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
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 
random.seed(seed)

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

# Fruit color mapping
FRUIT_COLORS = {
    'cherry': 'red',
    'pomegranate': 'red', 
    'strawberry': 'red',
    'tomato': 'red',
    'banana': 'yellow',
    'lemon': 'yellow',
    'corn': 'yellow',
    'broccoli': 'green',
    'cucumber': 'green',
    'brinjal': 'purple',
    'plum': 'purple',
    'orange': 'orange',
    'carrot': 'orange'
}

# All possible colors for evaluation
ALL_COLORS = ['yellow', 'red', 'green', 'purple', 'orange']


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def pad_to_square(image):
    """Pad image to square shape"""
    width, height = image.size
    max_dim = max(width, height)
    
    # Calculate padding for each side
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    
    # Apply padding (left, top, right, bottom)
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return torch_transforms.functional.pad(image, padding, fill=255, padding_mode='constant')

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Lambda(pad_to_square),
        torch_transforms.Resize(size, interpolation=interpolation),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


import torch

def compute_timestep_weight(t, T):
    """
    Compute weighting function w_t := exp(-7t) where t is normalized to [0, 1]
    Args:
        t: timestep value (integer or tensor)
        T: total number of timesteps (e.g., 1000)
    Returns:
        weight: exp(-7 * t_normalized) as a tensor
    """
    t = torch.tensor(t, dtype=torch.float16)  # ensure tensor
    T = torch.tensor(T, dtype=torch.float16)
    t_normalized = t / T
    weight = torch.exp(-7.0 * t_normalized)
    return weight



class CABDataset:
    """Custom dataset for CAB fruit images"""
    def __init__(self, root_dir, mode='compound', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'compound':
            self.image_paths = glob.glob(osp.join(root_dir, "fruit_combinations", "*.jpg"))
        else:
            self.image_paths = glob.glob(osp.join(root_dir, "single_images", "*.jpg"))
        
        print(f"Found {len(self.image_paths)} {mode} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        filename = osp.basename(image_path)
        name_part = filename.replace('.jpg', '').rsplit('_', 1)[0]
        
        if self.mode == 'compound':
            fruits = name_part.split('_')
        else:
            fruits = [name_part]
            
        return image, fruits, image_path


class DiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0
        self.other_fruit_predictions = 0
        self.other_mistakes = 0
        self.total_classifications = 0
        
        # Confusion matrix for single mode
        if self.args.mode == 'single':
            self.confusion_matrix = np.zeros((len(ALL_COLORS), len(ALL_COLORS)), dtype=int)
            self.color_to_idx = {color: idx for idx, color in enumerate(ALL_COLORS)}
        
        self._setup_models()
        self._setup_dataset()
        self._setup_noise()
        self._setup_run_folder()
        
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
        self.target_dataset = CABDataset(self.args.cab_folder, mode=self.args.mode, transform=transform)
        
    def _setup_noise(self):
        if self.args.noise_path is not None:
            assert not self.args.zero_noise
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        name = f"CAB_allcolors_{self.args.mode}_v{self.args.version}_{self.args.n_trials}trials_"
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
        # Add weighted suffix
        name += '_weighted'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, 'CAB_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'CAB', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def create_color_prompts(self, target_fruit):
        """Create prompts for all 5 colors for a specific fruit"""
        prompts = []
        for color in ALL_COLORS:
            # f"In this picture, the color of the {target_fruit} is {color}."
            prompt = f"A {color} {target_fruit}."
            prompts.append(prompt)
        return prompts

    def classify_prediction(self, predicted_color, correct_color, other_fruit_color=None):
        """Classify the type of prediction made"""
        if predicted_color == correct_color:
            return 'correct'
        elif other_fruit_color is not None and predicted_color == other_fruit_color:
            return 'other_fruit'
        else:
            return 'other_mistake'

    def eval_error(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                   text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2', T=1000):
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
                
                # Apply timestep weighting: w_t = exp(-7 * t_normalized)
                # weights = torch.stack([compute_timestep_weight(t.item(), T) for t in batch_ts]).to(error.device)
                # weighted_error = error * weights

                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors

    def eval_error_visualize(self, unet, scheduler, vae, original_image, latent, all_noise, ts, noise_idxs,
                text_embeds, text_embed_idxs, target_fruit, batch_size=32, dtype='float32', loss='l2', T=1000,
                visualize=False, prompts=None):
        """
        Evaluate denoising error in pixel space with comprehensive visualization.
        
        Args:
            prompts: List of prompt strings corresponding to text_embeds (needed for visualization titles)
        """
        assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
        pred_errors = torch.zeros(len(ts), device='cpu')
        idx = 0
        
        # Store data for visualization - organized by prompt
        if visualize:
            viz_data = defaultdict(lambda: {'images': [], 'errors': [], 'timesteps': []})
        
        with torch.inference_mode():
            for batch_idx in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[idx: idx + batch_size])
                batch_text_idxs = text_embed_idxs[idx: idx + batch_size]
                noise = all_noise[noise_idxs[idx: idx + batch_size]]
                
                # Create noised latent
                alpha_prod_t = scheduler.alphas_cumprod[batch_ts]
                sqrt_alpha_prod = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1).to(self.device)
                sqrt_one_minus_alpha_prod = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1).to(self.device)
                
                noised_latent = latent * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
                
                # Prepare inputs
                t_input = batch_ts.to(self.device).half() if dtype == 'float16' else batch_ts.to(self.device)
                text_input = text_embeds[batch_text_idxs]
                if dtype == 'float16':
                    text_input = text_input.half()
                    noised_latent = noised_latent.half()
                
                # Predict noise
                noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                
                # Denoise to get predicted clean latent
                denoised_latent = (noised_latent - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                
                reconstructed_image = vae.decode(denoised_latent / vae.config.scaling_factor).sample
            
                # Convert to float32 for error computation if needed
                if dtype == 'float16':
                    reconstructed_image = reconstructed_image.float()
                
                # Compute error in pixel space
                if loss == 'l2':
                    error = F.mse_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                elif loss == 'l1':
                    error = F.l1_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                elif loss == 'huber':
                    error = F.huber_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                else:
                    raise NotImplementedError
                
                # Average over spatial dimensions for the metric
                pred_errors[idx: idx + len(batch_ts)] = error.mean(dim=(1, 2)).detach().cpu()
                
                # Store samples for visualization - group by prompt
                if visualize:
                    for i in range(len(batch_ts)):
                        prompt_idx = batch_text_idxs[i]
                        viz_data[prompt_idx]['images'].append(reconstructed_image[i].detach().cpu())
                        viz_data[prompt_idx]['errors'].append(error[i].detach().cpu())
                        viz_data[prompt_idx]['timesteps'].append(batch_ts[i].item())
                
                idx += len(batch_ts)
        
        # Create visualizations for all prompts
        if visualize and len(viz_data) > 0:
            timestamp = datetime.now().strftime("%H%M%S")
            
            for prompt_idx, data in viz_data.items():
                prompt_text = prompts[prompt_idx] if prompts is not None else f"Prompt {prompt_idx}"
                
                # Convert lists to tensors
                recon_images = torch.stack(data['images'])
                error_maps = torch.stack(data['errors'])
                timesteps = torch.tensor(data['timesteps'])
                
                save_path = osp.join(self.run_folder, f'error_heatmap_prompt{prompt_idx}_{timestamp}.png')
                self.visualize_error_heatmap(
                    original_image, target_fruit, recon_images, error_maps, 
                    timesteps, prompt_text, save_path=save_path
                )
        
        return pred_errors


    def visualize_error_heatmap(self, original_image, target_fruit, reconstructed_images, 
                            error_maps, timesteps, prompt_text, save_path=None):
        """
        Visualize original image, reconstructed images, and error heatmaps for all timesteps.

        Args:
            original_image: Original image tensor (B, C, H, W)
            target_fruit: Name of target fruit
            reconstructed_images: Reconstructed images tensor (N, C, H, W) - all timesteps
            error_maps: Error maps tensor (N, H, W)
            timesteps: Timesteps for each image (N,)
            prompt_text: The prompt text being evaluated
            save_path: Path to save the figure
        """
        num_samples = len(reconstructed_images)
        
        # Sort by timestep for better visualization
        sort_idx = torch.argsort(timesteps)
        timesteps = timesteps[sort_idx]
        reconstructed_images = reconstructed_images[sort_idx]
        error_maps = error_maps[sort_idx]
        
        # Determine grid layout
        max_cols = 6  # Maximum images per row
        num_rows = (num_samples + max_cols - 1) // max_cols
        num_cols = min(num_samples, max_cols)
        
        fig = plt.figure(figsize=(5 * num_cols, 10 * num_rows))
        gs = fig.add_gridspec(num_rows + 1, num_cols, hspace=0.3, wspace=0.3)
        
        # Convert original image to numpy and normalize to [0, 1]
        orig_img = ((original_image[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1).astype(np.float32)
        
        # Plot original image spanning first row
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f'Original Image: {target_fruit}\nPrompt: "{prompt_text}"', 
                        fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Plot reconstructed images and error maps
        for i in range(num_samples):
            row = (i // num_cols) + 1
            col = i % num_cols
            
            # Create subplot with 2 rows for this position
            ax = fig.add_subplot(gs[row, col])
            
            recon_img = ((reconstructed_images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1).astype(np.float32)
            error_map = error_maps[i].cpu().numpy().astype(np.float32)
            
            # Create combined visualization: reconstructed on top, error heatmap on bottom
            combined_height = recon_img.shape[0] + error_map.shape[0]
            ax.clear()
            
            # Show reconstructed image
            ax.imshow(recon_img, extent=[0, 1, 0.5, 1])
            
            # Show error heatmap
            im = ax.imshow(error_map, cmap='hot', interpolation='nearest', 
                        extent=[0, 1, 0, 0.5], aspect='auto')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f't={timesteps[i].item()}\nAvg Error: {error_map.mean():.4f}', 
                        fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Reconstruction Error Analysis Across All Timesteps\nTarget: {target_fruit}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            plt.close()
        
        return fig


    # Update the eval_prob_adaptive method to pass prompts
    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, image, target_fruit, 
                        latent_size=64, all_noise=None, prompts=None):
        """
        Args:
            prompts: List of prompt strings for visualization titles
        """
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']
        max_n_samples = max(args.n_samples)

        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), 
                                device=latent.device)
        if args.dtype == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        data = dict()
        t_evaluated = set()
        remaining_prmpt_idxs = list(range(len(text_embeds)))
        start = T // max_n_samples // 2

        t_to_eval = np.linspace(800, 999, max_n_samples, dtype=int).tolist()
        
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
            
            # Pass prompts to eval_error for visualization
            if args.visualize:
                pred_errors = self.eval_error_visualize(
                    unet, scheduler, self.vae, image, latent, all_noise, ts, noise_idxs,
                    text_embeds, text_embed_idxs, target_fruit, args.batch_size, args.dtype, 
                    args.loss, T, visualize=False, prompts=prompts
                )
            else:
                pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                    text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss, T)
            
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


    # Update the classification methods to pass prompts
    def perform_color_classification_compound(self, image, fruit1, fruit2, target_fruit):
        """Perform color classification for a specific fruit in compound image"""
        prompts = self.create_color_prompts(target_fruit)
        
        text_input = self.tokenizer(prompts, padding="max_length",
                            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.inference_mode():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            img_input = image.to(self.device).unsqueeze(0)
            if self.args.dtype == 'float16':
                img_input = img_input.half()
            x0 = self.vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215

        all_losses, pred_idx, pred_errors = self.eval_prob_adaptive(
            self.unet, x0, text_embeddings, self.scheduler, 
            self.args, img_input, target_fruit, self.latent_size, self.all_noise,
            prompts=prompts  # Pass prompts here
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[target_fruit.lower()]
        other_fruit = fruit2 if target_fruit == fruit1 else fruit1
        other_fruit_color = FRUIT_COLORS[other_fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, other_fruit_color)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx

    def perform_color_classification_single(self, image, fruit):
        """Perform color classification for a single fruit image"""
        prompts = self.create_color_prompts(fruit)
        
        text_input = self.tokenizer(prompts, padding="max_length",
                            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.inference_mode():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            img_input = image.to(self.device).unsqueeze(0)
            if self.args.dtype == 'float16':
                img_input = img_input.half()
            x0 = self.vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215

        all_losses, pred_idx, pred_errors = self.eval_prob_adaptive(
            self.unet, x0, text_embeddings, self.scheduler, 
            self.args, image, fruit, self.latent_size, self.all_noise, prompts=prompts
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx

    def save_confusion_matrix(self):
        """Save confusion matrix visualization and CSV for single mode"""
        if self.args.mode != 'single':
            return
        
        # Save as CSV
        df = pd.DataFrame(self.confusion_matrix, index=ALL_COLORS, columns=ALL_COLORS)
        csv_path = osp.join(self.run_folder, 'confusion_matrix.csv')
        df.to_csv(csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ALL_COLORS, yticklabels=ALL_COLORS,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Color')
        plt.ylabel('True Color')
        plt.title('Color Classification Confusion Matrix (Weighted)')
        plt.tight_layout()
        
        # Save figure
        fig_path = osp.join(self.run_folder, 'confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix visualization saved to {fig_path}")
        
        # Calculate and save per-class metrics
        metrics_path = osp.join(self.run_folder, 'per_class_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Per-Class Metrics (Weighted)\n")
            f.write("=" * 50 + "\n\n")
            
            for i, color in enumerate(ALL_COLORS):
                true_positives = self.confusion_matrix[i, i]
                false_positives = self.confusion_matrix[:, i].sum() - true_positives
                false_negatives = self.confusion_matrix[i, :].sum() - true_positives
                total_true = self.confusion_matrix[i, :].sum()
                
                if total_true > 0:
                    recall = true_positives / total_true * 100
                else:
                    recall = 0.0
                
                if (true_positives + false_positives) > 0:
                    precision = true_positives / (true_positives + false_positives) * 100
                else:
                    precision = 0.0
                
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                f.write(f"{color.upper()}:\n")
                f.write(f"  Total samples: {total_true}\n")
                f.write(f"  Correct predictions: {true_positives}\n")
                f.write(f"  Precision: {precision:.2f}%\n")
                f.write(f"  Recall: {recall:.2f}%\n")
                f.write(f"  F1-Score: {f1:.2f}\n\n")
        
        print(f"Per-class metrics saved to {metrics_path}")

    def save_results_summary(self):
        """Save a summary of color classification results."""
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"CAB Color Diffusion Classification Results Summary ({self.args.mode.title()} Mode)\n")
            f.write(f"Using Timestep Weighting: w_t = exp(-7t)\n")
            f.write("==========================================================\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                other_fruit_acc = (self.other_fruit_predictions / self.total_classifications * 100)
                other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                
                f.write(f"Color Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.args.mode == 'compound':
                    f.write(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
                    f.write(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                else:
                    f.write(f"Incorrect predictions: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            if self.args.mode == 'compound':
                f.write(f"Expected for {len(self.target_dataset)} compound images: {len(self.target_dataset) * 2} classifications\n\n")
            else:
                f.write(f"Expected for {len(self.target_dataset)} single images: {len(self.target_dataset)} classifications\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"Mode: {self.args.mode}\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Loss function: {self.args.loss}\n")
            f.write(f"Data type: {self.args.dtype}\n")
            f.write(f"Interpolation: {self.args.interpolation}\n")
            f.write(f"Timestep weighting: exp(-7t)\n")
            f.write(f"Adaptive sampling - n_samples: {self.args.n_samples}\n")
            f.write(f"Adaptive sampling - to_keep: {self.args.to_keep}\n\n")
            
            f.write(f"All possible colors evaluated: {', '.join(ALL_COLORS)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation_compound(self):
        """Run evaluation for compound images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = get_formatstr(len(self.target_dataset) * 2 - 1)
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.1f}%, Other Fruit: {other_fruit_acc:.1f}% ({self.total_classifications})')
            
            image, fruits, image_path = self.target_dataset[i]
            
            if len(fruits) != 2:
                print(f"Skipping {image_path}: Expected 2 fruits, got {len(fruits)}")
                continue
                
            fruit1, fruit2 = fruits
            if fruit1.lower() not in FRUIT_COLORS or fruit2.lower() not in FRUIT_COLORS:
                print(f"Skipping {image_path}: Unknown fruits {fruit1}, {fruit2}")
                continue
            
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {classification_idx}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                predicted_color, correct_color, prediction_type, prompts, pred_idx = self.perform_color_classification_compound(
                    image, fruit1, fruit2, target_fruit
                )
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_color=predicted_color,
                    correct_color=correct_color,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=target_fruit,
                    fruits=[fruit1, fruit2],
                    prompts=prompts,
                    image_path=image_path,
                    classification_idx=classification_idx
                ), fname)

    def run_evaluation_single(self):
        """Run evaluation for single images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = get_formatstr(len(self.target_dataset) - 1)
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.2f}% ({self.correct_predictions}/{self.total_classifications})')
            
            image, fruits, image_path = self.target_dataset[i]
            
            if len(fruits) != 1:
                print(f"Skipping {image_path}: Expected 1 fruit, got {len(fruits)}")
                continue
                
            fruit = fruits[0]
            if fruit.lower() not in FRUIT_COLORS:
                print(f"Skipping {image_path}: Unknown fruit {fruit}")
                continue
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                print(f'Skipping classification {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_color = data['predicted_color']
                    correct_color = data['correct_color']
                    
                    # Update confusion matrix
                    true_idx = self.color_to_idx[correct_color]
                    pred_idx = self.color_to_idx[predicted_color]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.other_mistakes += 1
                    self.total_classifications += 1
                continue
            
            predicted_color, correct_color, prediction_type, prompts, pred_idx = self.perform_color_classification_single(
                image, fruit
            )
            
            # Update confusion matrix
            true_idx = self.color_to_idx[correct_color]
            pred_idx_cm = self.color_to_idx[predicted_color]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_color=predicted_color,
                correct_color=correct_color,
                prediction_type=prediction_type,
                pred_idx=pred_idx,
                target_fruit=fruit,
                fruits=[fruit],
                prompts=prompts,
                image_path=image_path,
                classification_idx=i
            ), fname)

    def run_evaluation(self):
        """Run evaluation based on mode"""
        if self.args.mode == 'compound':
            self.run_evaluation_compound()
        else:
            self.run_evaluation_single()
        
        # Generate summary
        self.save_results_summary()
        
        # Save confusion matrix for single mode
        if self.args.mode == 'single':
            self.save_confusion_matrix()
        
        # Print final results
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Color Classification Results ({self.args.mode.title()} Mode, Weighted):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.args.mode == 'compound':
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                other_mistakes_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)")
                print(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)")
            else:
                incorrect_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Incorrect predictions: {self.other_mistakes} ({incorrect_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--cab_folder', type=str,
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/CAB',
                        help='Path to CAB folder containing fruit_combinations and single_images folders')
    parser.add_argument('--mode', type=str, default='compound', choices=['compound', 'single'],
                        help='Mode: compound for two-fruit images, single for single-fruit images')
    
    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', default=[1], type=int)
    parser.add_argument('--n_samples', nargs='+', default=[50], type=int)

    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize error heatmaps during evaluation')

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # Create evaluator and run
    evaluator = DiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()