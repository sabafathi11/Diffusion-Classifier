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

# DAAM imports
from daam import set_seed, trace
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

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


class AttentionDiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        if self.args.prompt_path is not None:
            self._setup_prompts()
        self._setup_run_folder()
        
        # Set up DAAM pipeline
        self._setup_daam_pipeline()
        
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
        
        # Get text embeddings for all prompts
        text_input = self.tokenizer(self.prompts_df.classname.tolist(), padding="max_length",
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
        
    def _setup_run_folder(self):
        # make run output folder
        name = f"attention_v{self.args.version}_"
        name += f"t{self.args.attention_timesteps}_"
        name += f"thresh{self.args.attention_threshold}"
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

    def _setup_daam_pipeline(self):
        """Setup DAAM pipeline for attention visualization"""
        model_id = f"stabilityai/stable-diffusion-{self.args.version}"
        self.daam_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.args.dtype == 'float16' else torch.float32
        ).to(self.device)
        
    def get_attention_scores(self, image_tensor, prompts):
        """
        Get attention scores for each prompt on the given image
        
        Args:
            image_tensor: preprocessed image tensor [1, 3, H, W] 
            prompts: list of prompt strings
            
        Returns:
            attention_scores: list of attention scores for each prompt
        """
        # Convert tensor back to PIL for DAAM
        # Denormalize: from [-1, 1] back to [0, 1]
        img_denorm = (image_tensor.squeeze(0) + 1) / 2
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to PIL
        img_pil = torch_transforms.ToPILImage()(img_denorm.cpu())
        
        attention_scores = []
        
        for prompt in prompts:
            try:
                # Use DAAM to generate and trace attention
                set_seed(42)  # For reproducibility
                
                with trace(self.daam_pipe) as tc:
                    # Generate image with current prompt 
                    # We use the original image as init_image for img2img-like behavior
                    out = self.daam_pipe(
                        prompt,
                        num_inference_steps=self.args.attention_timesteps,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
                    
                # Get attention maps
                # We want to see how much the prompt attends to the input
                # Extract class name from prompt (assuming format like "a photo of a {class}")
                words = prompt.split()
                
                # Try to identify the main noun/class word
                # This is dataset dependent - you might need to adjust this logic
                if "of a" in prompt:
                    class_word_idx = words.index("a") + 1
                    if class_word_idx < len(words):
                        class_word = words[class_word_idx]
                    else:
                        class_word = words[-1]  # fallback to last word
                else:
                    class_word = words[-1]  # fallback to last word
                    
                # Get attention map for the class word
                attention_map = tc.compute_global_heat_map()
                
                # Compute attention score as mean activation above threshold
                if self.args.attention_method == 'mean':
                    score = attention_map.mean().item()
                elif self.args.attention_method == 'max':
                    score = attention_map.max().item()
                elif self.args.attention_method == 'above_threshold':
                    score = (attention_map > self.args.attention_threshold).float().mean().item()
                elif self.args.attention_method == 'weighted_sum':
                    # Weight by attention values above threshold
                    mask = attention_map > self.args.attention_threshold
                    score = (attention_map * mask).sum().item()
                else:
                    score = attention_map.mean().item()
                    
                attention_scores.append(score)
                
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
                attention_scores.append(0.0)  # fallback score
                
        return attention_scores
    
    def classify_by_attention(self, image, prompts_to_use):
        """
        Classify image based on attention scores
        
        Args:
            image: preprocessed image tensor
            prompts_to_use: list of prompts/class names to evaluate
            
        Returns:
            pred_class_idx: predicted class index
            all_scores: attention scores for all prompts
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Get attention scores for all prompts
        attention_scores = self.get_attention_scores(image, prompts_to_use)
        
        # Find the prompt/class with highest attention score
        pred_class_idx = np.argmax(attention_scores)
        
        return pred_class_idx, attention_scores

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
            
            prompts_to_use = self.prompts_df.classname.tolist()
            
            # Classify based on attention
            pred_idx, attention_scores = self.classify_by_attention(image, prompts_to_use)
            
            pred = self.prompts_df.classidx.iloc[pred_idx]
                
            # Save results
            torch.save({
                'attention_scores': attention_scores,
                'pred': pred,
                'label': label,
                'pred_idx': pred_idx
            }, fname)
            
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

    # model args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')

    # attention-specific args
    parser.add_argument('--attention_timesteps', type=int, default=20, help='Number of timesteps for attention computation')
    parser.add_argument('--attention_threshold', type=float, default=0.1, help='Threshold for attention map')
    parser.add_argument('--attention_method', type=str, default='mean', 
                        choices=['mean', 'max', 'above_threshold', 'weighted_sum'],
                        help='Method to compute attention scores')

    # data args
    parser.add_argument('--prompt_path', type=str, default=None, help='Path to csv file with prompts to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--samples_per_class', type=int, default=None, help='Number of samples per class for balanced subset')

    # run args
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = AttentionDiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()