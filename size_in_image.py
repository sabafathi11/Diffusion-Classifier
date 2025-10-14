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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForImageSegmentation
from PIL import Image
from pathlib import Path
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 43

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 

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


class MultiObjectDataset:
    """Custom dataset for multi-object images with filename-based labels."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.root_dir.glob("*.png")) + list(self.root_dir.glob("*.jpg")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # Extract all unique objects from all filenames
        all_objects_set = set()
        for img_path in self.image_paths:
            filename = img_path.stem
            objects = filename.split('_')
            all_objects_set.update(objects)
        
        self.all_objects = sorted(list(all_objects_set))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        print(f"Found {len(self.all_objects)} unique objects: {self.all_objects}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Extract objects from filename (e.g., "axe_backpack_bird.png" -> ["axe", "backpack", "bird"])
        filename = img_path.stem
        objects = filename.split('_')
        
        return image, objects, filename



def create_prompts_for_scenario(dataset, scenario_mode):
    """
    Create positive and negative prompts for each image.
    
    Scenario 1: Biggest object first in both prompts, wrong object at position 3 in negative
    Scenario 2: Biggest object last in positive, first in negative; wrong object at position 3 in negative
    """
    prompt_data = []
    all_objects = dataset.all_objects
    
    for idx in range(len(dataset)):
        _, objects, filename = dataset[idx]
        
        if len(objects) < 3:
            print(f"Warning: {filename} has less than 3 objects, skipping")
            continue
        
        obj1, obj2, obj3 = objects[0], objects[1], objects[2]  # obj1 is biggest
        
        # Select a wrong object that is NOT in the current image
        available_wrong_objects = [obj for obj in all_objects if obj not in objects]
        
        if not available_wrong_objects:
            print(f"Warning: {filename} - no available wrong objects, skipping")
            continue
        
        wrong_object = random.choice(available_wrong_objects)
        
        if scenario_mode == 1:
            # Scenario 1: Biggest object first in both
            positive_prompt = f"a photo of a {obj1}, a {obj2}, and a {obj3}"
            negative_prompt = f"a photo of a {obj1}, a {obj2}, and a {wrong_object}"
            
        elif scenario_mode == 2:
            # Scenario 2: Biggest object last in positive, first in negative
            positive_prompt = f"a photo of a {obj2}, a {obj3}, and a {obj1}"
            negative_prompt = f"a photo of a {obj1}, a {obj2}, and a {wrong_object}"
        
        else:
            raise ValueError(f"Unknown scenario mode: {scenario_mode}")
        
        print(f"Image: {filename}"
              f"\n  Positive: {positive_prompt}"
              f"\n  Negative: {negative_prompt}"
              f"\n  Wrong object: {wrong_object}\n")
        
        prompt_data.append({
            'filename': filename,
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'objects': '_'.join(objects),
            'biggest_object': obj1,
            'wrong_object': wrong_object
        })
    
    return pd.DataFrame(prompt_data)


class BackgroundRemover:
    """Wrapper for BiRefNet background removal."""
    
    def __init__(self, device="cuda"):
        print("Loading BiRefNet for background removal...")
        torch.set_float32_matmul_precision("high")
        
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(device)
        
        self.transform_image = torch_transforms.Compose([
            torch_transforms.Resize((1024, 1024)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.device = device
        print("BiRefNet loaded successfully")
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from a PIL image and replace with white."""
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = torch_transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        white_background = Image.new('RGB', image_size, (255, 255, 255))
        white_background.paste(image, mask=mask)
        
        return white_background


class MultiObjectDiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.all_predictions = []  # Will store 0 for positive, 1 for negative
        self.all_labels = []  # All should be 0 (positive is correct)
        self.results_details = []
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        self._setup_run_folder()
        self._setup_prompts()
        self._setup_noise()
        
        # Setup background remover if requested
        if self.args.remove_background:
            self.bg_remover = BackgroundRemover(device=self.device)
        else:
            self.bg_remover = None
        
    def _setup_models(self):
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = get_sd_model(self.args)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        torch.backends.cudnn.benchmark = True
        
    def _setup_dataset(self):
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        self.latent_size = self.args.img_size // 8
        self.target_dataset = MultiObjectDataset(self.args.dataset_path, transform=transform)
        
    def _setup_prompts(self):
        # Create prompts based on scenario mode
        self.prompts_df = create_prompts_for_scenario(
            self.target_dataset, 
            self.args.scenario_mode,
        )
        
        # Save prompts for reference
        prompts_save_path = osp.join(self.run_folder, 'prompts.csv')
        self.prompts_df.to_csv(prompts_save_path, index=False)
        print(f"Prompts saved to {prompts_save_path}")
        
        # Encode prompts
        all_prompts = []
        for _, row in self.prompts_df.iterrows():
            all_prompts.append(row['positive_prompt'])
            all_prompts.append(row['negative_prompt'])
        
        text_input = self.tokenizer(all_prompts, padding="max_length",
                            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_input.input_ids), 100):
                text_embeddings = self.text_encoder(
                    text_input.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        self.text_embeddings = torch.cat(embeddings, dim=0)
        assert len(self.text_embeddings) == len(all_prompts)
        
    def _setup_noise(self):
        if self.args.noise_path is not None:
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        name = f"scenario{self.args.scenario_mode}_v{self.args.version}_{self.args.n_trials}trials_"
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
        if self.args.remove_background:
            name += '_nobg'
        if self.args.extra is not None:
            name += f'_{self.args.extra}'
        
        self.run_folder = osp.join(LOG_DIR, 'multi_object', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')
    
    def process_image_tensor(self, image_tensor):
        """Convert tensor to PIL, remove background, convert back to tensor."""
        image_pil = torch_transforms.ToPILImage()(image_tensor * 0.5 + 0.5)
        image_nobg = self.bg_remover.remove_background(image_pil)
        bg = Image.new('RGB', image_nobg.size, (255, 255, 255))
        bg.paste(image_nobg, mask=image_nobg.split()[3] if image_nobg.mode == 'RGBA' else None)
        image_tensor_out = torch_transforms.ToTensor()(bg)
        image_tensor_out = torch_transforms.Normalize([0.5], [0.5])(image_tensor_out)
        return image_tensor_out
    
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
            
            for prompt_i in remaining_prmpt_idxs:
                mask = torch.tensor(text_embed_idxs) == prompt_i
                prompt_ts = torch.tensor(ts)[mask]
                prompt_pred_errors = pred_errors[mask]
                if prompt_i not in data:
                    data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
                else:
                    data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                    data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

            errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]
        
        all_losses = torch.zeros(len(text_embeds))
        for prompt_i in data:
            all_losses[prompt_i] = data[prompt_i]['pred_errors'].mean()

        return all_losses, pred_idx, data

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
                if dtype == 'float16':
                    text_input = text_input.half()
                    noised_latent = noised_latent.half()
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

    def save_results_summary(self):
        """Save comprehensive summary of results."""
        if len(self.all_predictions) == 0:
            print("No results to summarize")
            return
            
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Multi-Object Diffusion Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            correct = sum(1 for p in self.all_predictions if p == 0)  # 0 = positive (correct)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Scenario Mode: {self.args.scenario_mode}\n")
            if self.args.scenario_mode == 1:
                f.write("  - Positive: biggest object first\n")
                f.write("  - Negative: biggest object first, wrong 3rd object\n")
            else:
                f.write("  - Positive: biggest object last\n")
                f.write("  - Negative: biggest object first, wrong 3rd object\n")
            
            f.write(f"Overall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct (chose positive prompt): {correct}\n")
            f.write(f"Incorrect (chose negative prompt): {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Background removal: {self.args.remove_background}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Loss function: {self.args.loss}\n")
        
        print(f"Results summary saved to {summary_path}")
        
        # Save detailed results
        results_df = pd.DataFrame(self.results_details)
        results_csv_path = osp.join(self.run_folder, 'detailed_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to {results_csv_path}")

    def run_evaluation(self):
        formatstr = get_formatstr(len(self.target_dataset) - 1)
        correct = 0
        total = 0
        pbar = tqdm.tqdm(range(len(self.target_dataset)))
        
        for i in pbar:
            if total > 0:
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print(f'Skipping {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['chose_positive'])
                    total += 1
                    self.all_predictions.append(0 if data['chose_positive'] else 1)
                    self.all_labels.append(0)
                continue
            
            image, objects, filename = self.target_dataset[i]
            
            # Remove background if requested
            if self.args.remove_background:
                image = self.process_image_tensor(image)
            
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215

            # Get embeddings for this image's positive and negative prompts
            # Each image has 2 prompts: positive (idx*2) and negative (idx*2+1)
            text_embeddings_pair = self.text_embeddings[i*2:(i+1)*2]
                
            _, pred_idx, pred_errors = self.eval_prob_adaptive(
                self.unet, x0, text_embeddings_pair, self.scheduler, 
                self.args, self.latent_size, self.all_noise
            )
            
            # pred_idx is 0 for positive, 1 for negative
            chose_positive = (pred_idx == 0)
            
            result_detail = {
                'filename': filename,
                'objects': '_'.join(objects),
                'chose_positive': chose_positive,
                'positive_loss': pred_errors[pred_idx]['pred_errors'].mean().item(),
                'negative_loss': pred_errors[1-pred_idx]['pred_errors'].mean().item(),
                'positive_prompt': self.prompts_df.iloc[i]['positive_prompt'],
                'negative_prompt': self.prompts_df.iloc[i]['negative_prompt']
            }
            self.results_details.append(result_detail)
            
            torch.save({
                'errors': pred_errors,
                'chose_positive': chose_positive,
                'filename': filename,
                'objects': objects
            }, fname)
            
            if chose_positive:
                correct += 1
            total += 1
            
            self.all_predictions.append(0 if chose_positive else 1)
            self.all_labels.append(0)  # Ground truth is always 0 (positive)
        
        self.save_results_summary()


def main():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/size-in-image',
                        help='Path to dataset directory')
    parser.add_argument('--scenario_mode', type=int, default=2, choices=[1, 2],
                        help='Scenario 1: biggest first in both; Scenario 2: biggest last in positive, first in negative')

    # Model args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'))
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='Extra string for run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'))
    parser.add_argument('--remove_background', action='store_true', default=False, 
                        help='Remove background using BiRefNet')
    parser.add_argument('--run_folder', type=str, default=None, help='If set, use this folder to save results')

    # Adaptive sampling args
    parser.add_argument('--to_keep', nargs='+', type=int, default=[1])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[50])

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    evaluator = MultiObjectDiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()