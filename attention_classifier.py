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


# Import DAAM components
from daam import trace, set_seed
from diffusers import StableDiffusionXLImg2ImgPipeline
from daam.heatmap import GlobalHeatMap

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


class DAAMDiffusionClassifier:
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
        # Create diffusion pipeline for DAAM using DiffusionPipeline like in the repo example
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'  # You may want to make this configurable
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16'
        ).to(device)
        
    def _setup_dataset(self):
        # set up dataset
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        self.latent_size = self.args.img_size // 8
        self.target_dataset = get_target_dataset(self.args.dataset, train=self.args.split == 'train', transform=transform)
        
    def _setup_prompts(self):
        self.prompts_df = pd.read_csv(self.args.prompt_path)
        
        # Use classname column for DAAM processing
        self.class_names = self.prompts_df.classname.tolist()
        
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
        name = f"daam_v{self.args.version}_{self.args.n_trials}trials"
        if self.args.interpolation != 'bicubic':
            name += f'_{self.args.interpolation}'
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



    def compute_daam_scores(self, image, class_names):
        """
        Compute DAAM heat map scores for each candidate class name.
        Returns the total activation (sum of heat map values) for each class.
        """
        # Convert tensor image back to PIL for DAAM processing
        if isinstance(image, torch.Tensor):
            # Denormalize and convert to PIL
            image_pil = torch_transforms.ToPILImage()(image * 0.5 + 0.5)
        else:
            image_pil = image
        
        daam_scores = []

        prompt = ' '.join(class_names)

        for class_name in class_names:
            try:
                gen = set_seed(self.args.daam_seed)  # for reproducibility
                with torch.no_grad():
                    with trace(self.pipe) as tc:
                        # Generate using the class name as prompt
                        out = self.pipe(
                            prompt=prompt,
                            image=image_pil,
                            strength=0.03,      # tiny value to avoid changes but keep pipeline working
                            num_inference_steps=self.args.daam_steps,
                            generator=gen
                        )
                            
                        # Compute global heat map
                        global_word_heat_map = tc.compute_global_heat_map()
                        word_heat_map = global_word_heat_map.compute_word_heat_map(class_name)
                            
                        # Get the total activation as classification score
                        heat_map_tensor = word_heat_map.heatmap

                        # total_activation = float(heat_map_tensor.sum().item())

                        threshold = 0.1
                        attended_pixels = (heat_map_tensor >= threshold).sum()

                        daam_scores.append(attended_pixels)
                            
            except Exception as e:
                print(f"Error processing class '{class_name}': {e}")
                daam_scores.append(0.0)  # Default to 0 if error
                    
        return torch.tensor(daam_scores)

    def eval_daam_classification(self, image):
        """
        Evaluate using DAAM heat maps for all classes at once.
        Returns DAAM scores for all classes and the predicted class index.
        """
        # Compute DAAM scores for all class names
        daam_scores = self.compute_daam_scores(image, self.class_names)

        print('\n\n\n\n\n')
        print(self.class_names)
        print(daam_scores)
        
        # Find the class with highest activation
        pred_idx = torch.argmin(daam_scores).item()
        
        return daam_scores, pred_idx

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
                pbar.set_description(f'DAAM Acc: {100 * correct / total:.2f}%')
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['pred'] == data['label'])
                    total += 1
                continue
                
            image, label = self.target_dataset[i]

            print(self.prompts_df[self.prompts_df['classidx'] == label]['classname'].values[0])
            
            # Compute DAAM-based classification scores for all classes
            daam_scores, pred_idx = self.eval_daam_classification(image)
            
            # Get predicted class from CSV mapping
            pred = self.prompts_df.classidx[pred_idx]
                
            torch.save(dict(daam_scores=daam_scores, pred=pred, label=label), fname)
            if pred == label:
                correct += 1
            total += 1

        print(f'Final DAAM Classification Accuracy: {100 * correct / total:.2f}%')


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, default=None, help='Path to csv file with prompts to use')
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
    # DAAM-specific args
    parser.add_argument('--daam_steps', type=int, default=100, help='Number of inference steps for DAAM')
    parser.add_argument('--daam_seed', type=int, default=0, help='Seed for DAAM generation')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = DAAMDiffusionClassifier(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()