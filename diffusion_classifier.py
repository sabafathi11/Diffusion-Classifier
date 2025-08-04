import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def create_balanced_subset(dataset, samples_per_class):
    """Create a balanced subset of the dataset with specified samples per class."""
    class_to_indices = defaultdict(list)
    
    # Group indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices[label].append(idx)
    
    balanced_indices = []
    
    # Sample from each class
    for class_label, indices in class_to_indices.items():
        if len(indices) >= samples_per_class:
            # If we have enough samples, randomly select
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            # If we don't have enough samples, take all available
            selected = indices
            print(f"Warning: Class {class_label} only has {len(indices)} samples, less than requested {samples_per_class}")
        
        balanced_indices.extend(selected.tolist() if hasattr(selected, 'tolist') else selected)
    
    return sorted(balanced_indices)


class DiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        if self.args.prompt_path is not None:
            self._setup_prompts()
        self._setup_noise()
        self._setup_run_folder()
        
    def _setup_models(self):
        # load pretrained models
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = get_sd_model(self.args)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        torch.backends.cudnn.benchmark = True
        
    def _setup_dataset(self):
        # set up dataset
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        self.latent_size = self.args.img_size // 8
        self.target_dataset = get_target_dataset(self.args.dataset, train=self.args.split == 'train', transform=transform)
        
    def _setup_prompts(self):
        self.prompts_df = pd.read_csv(self.args.prompt_path)
        
        # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
        text_input = self.tokenizer(self.prompts_df.prompt.tolist(), padding="max_length",
                               max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_input.input_ids), 100):
                text_embeddings = self.text_encoder(
                    text_input.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        self.text_embeddings = torch.cat(embeddings, dim=0)
        assert len(self.text_embeddings) == len(self.prompts_df)
        
    def _setup_noise(self):
        # load noise
        if self.args.noise_path is not None:
            assert not self.args.zero_noise
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        # make run output folder
        name = f"v{self.args.version}_{self.args.n_trials}trials_"
        name += '_'.join(map(str, self.args.to_keep)) + 'keep_'
        name += '_'.join(map(str, self.args.n_samples)) + 'samples'
        if self.args.interpolation != 'bicubic':
            name += f'_{self.args.interpolation}'
        if self.args.loss == 'l1':
            name += '_l1'
        elif self.args.loss == 'huber':
            name += '_huber'
        if self.args.img_size != 512:
            name += f'_{self.args.img_size}'
        if self.args.samples_per_class is not None:
            name += f'_{self.args.samples_per_class}spc'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset + '_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset, name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']
        max_n_samples = max(args.n_samples)

        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
        if args.dtype == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        data = dict()
        t_evaluated = set()
        remaining_prmpt_idxs = list(range(len(text_embeds)))
        start = T // max_n_samples // 2
        t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

        for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            for prompt_i in remaining_prmpt_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                    ts.extend([t] * args.n_trials)
                    noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_i] * args.n_trials)
            t_evaluated.update(curr_t_to_eval)
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                     text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
            # match up computed errors to the data
            for prompt_i in remaining_prmpt_idxs:
                mask = torch.tensor(text_embed_idxs) == prompt_i
                prompt_ts = torch.tensor(ts)[mask]
                prompt_pred_errors = pred_errors[mask]
                if prompt_i not in data:
                    data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
                else:
                    data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                    data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

            # compute the next remaining idxs
            errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

        # organize the output
        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]
        
        all_losses = torch.zeros(len(text_embeds))
        for prompt_i in data:
            all_losses[prompt_i] = data[prompt_i]['pred_errors'].mean()

        return all_losses,pred_idx, data

    def eval_error(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                   text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
        assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
        pred_errors = torch.zeros(len(ts), device='cpu')
        idx = 0
        with torch.inference_mode():
            for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[idx: idx + batch_size])
                noise = all_noise[noise_idxs[idx: idx + batch_size]]
                noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
                                noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
                t_input = batch_ts.to(self.device).half() if dtype == 'float16' else batch_ts.to(self.device)
                text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
                noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                if loss == 'l2':
                    error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'l1':
                    error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'huber':
                    error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                else:
                    raise NotImplementedError
                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors

    def run_evaluation(self):
        # subset of dataset to evaluate
        if self.args.subset_path is not None:
            idxs = np.load(self.args.subset_path).tolist()
        elif self.args.samples_per_class is not None:
            # Create balanced subset
            idxs = create_balanced_subset(self.target_dataset, self.args.samples_per_class)
            print(f'Created balanced subset with {len(idxs)} total samples')
        else:
            idxs = list(range(len(self.target_dataset)))
        idxs_to_eval = idxs[self.args.worker_idx::self.args.n_workers]

        formatstr = get_formatstr(len(self.target_dataset) - 1)
        correct = 0
        total = 0
        pbar = tqdm.tqdm(idxs_to_eval)
        for i in pbar:
            if total > 0:
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['pred'] == data['label'])
                    total += 1
                continue
            image, label = self.target_dataset[i]
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            _,pred_idx, pred_errors = self.eval_prob_adaptive(self.unet, x0, self.text_embeddings, self.scheduler, self.args, self.latent_size, self.all_noise)
            pred = self.prompts_df.classidx[pred_idx]
            torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
            if pred == label:
                correct += 1
            total += 1


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, default=False, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--samples_per_class', type=int, default=None, help='Number of samples per class for balanced subset')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # Create evaluator and run
    evaluator = DiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()