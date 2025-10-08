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
from datetime import datetime

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


def create_balanced_subset(dataset, samples_per_class):
    """Create a balanced subset of the dataset with specified samples per class."""
    class_to_indices = defaultdict(list)
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices[label].append(idx)
    
    balanced_indices = []
    
    for class_label, indices in class_to_indices.items():
        if len(indices) >= samples_per_class:
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected = indices
            print(f"Warning: Class {class_label} only has {len(indices)} samples, less than requested {samples_per_class}")
        
        balanced_indices.extend(selected.tolist() if hasattr(selected, 'tolist') else selected)
    
    return sorted(balanced_indices)


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
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create white background
        white_background = Image.new('RGB', image_size, (255, 255, 255))
        
        # Composite the image onto white background using the mask
        white_background.paste(image, mask=mask)
        
        return white_background

class DiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.all_predictions = []
        self.all_labels = []
        self.classidx_to_name = {}
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        self._setup_prompts()
        self._setup_noise()
        self._setup_run_folder()
        self._setup_class_names()
        
        # Setup background remover if requested
        if self.args.remove_background:
            self.bg_remover = BackgroundRemover(device=self.device)
        else:
            self.bg_remover = None
  
        
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
        if self.args.noise_path is not None:
            assert not self.args.zero_noise
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
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
        if self.args.remove_background:
            name += '_nobg'
        if self.args.visualize:
            name += '_viz'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset + '_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset, name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def _setup_class_names(self):
        """Setup class name mapping"""
        if hasattr(self.target_dataset, 'classes'):
            for idx, class_name in enumerate(self.target_dataset.classes):
                self.classidx_to_name[idx] = class_name
        elif self.args.prompt_path is not None and hasattr(self, 'prompts_df'):
            if 'class_name' in self.prompts_df.columns:
                for _, row in self.prompts_df.iterrows():
                    self.classidx_to_name[row['classidx']] = row['class_name']
        self.classnames = [self.classidx_to_name.get(i, str(i)) for i in range(len(self.classidx_to_name))]
    
    def process_image_tensor(self, image_tensor):
        """Convert tensor to PIL, remove background, convert back to tensor."""
        # Convert from tensor [-1, 1] to PIL
        image_pil = torch_transforms.ToPILImage()(image_tensor * 0.5 + 0.5)
        
        # Remove background
        image_nobg = self.bg_remover.remove_background(image_pil)
        
        # Convert back to RGB (paste on white background to handle transparency)
        bg = Image.new('RGB', image_nobg.size, (255, 255, 255))
        bg.paste(image_nobg, mask=image_nobg.split()[3] if image_nobg.mode == 'RGBA' else None)
        
        # Convert back to tensor [-1, 1]
        image_tensor_out = torch_transforms.ToTensor()(bg)
        image_tensor_out = torch_transforms.Normalize([0.5], [0.5])(image_tensor_out)
        
        return image_tensor_out

    def visualize_error_heatmap(self, original_image, true_label, reconstructed_images, 
                            error_maps, timesteps, prompt_text, save_path=None):
        """
        Visualize original image, reconstructed images, and error heatmaps for all timesteps.

        Args:
            original_image: Original image tensor (B, C, H, W)
            true_label: True class label
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
        
        # Get class name
        class_name = self.classidx_to_name.get(true_label, str(true_label))
        
        # Plot original image spanning first row
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f'Original Image: {class_name}\nPrompt: "{prompt_text}"', 
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
        
        plt.suptitle(f'Reconstruction Error Analysis Across Timesteps\nTrue Class: {class_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig

    def eval_error_visualize(self, unet, scheduler, vae, original_image, latent, all_noise, ts, noise_idxs,
                text_embeds, text_embed_idxs, true_label, batch_size=32, dtype='float32', loss='l2', T=1000,
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
                
                # Decode to pixel space
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
                    original_image, true_label, recon_images, error_maps, 
                    timesteps, prompt_text, save_path=save_path
                )
        
        return pred_errors
    
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

    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, original_image, true_label,
                        latent_size=64, all_noise=None, prompts=None):
        """
        Args:
            original_image: Original image tensor for visualization
            true_label: True class label
            prompts: List of prompt strings for visualization titles
        """
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
            
            # Use visualization version if requested
            if args.visualize:
                pred_errors = self.eval_error_visualize(
                    unet, scheduler, self.vae, original_image, latent, all_noise, ts, noise_idxs,
                    text_embeds, text_embed_idxs, true_label, args.batch_size, args.dtype, 
                    args.loss, T, visualize=True, prompts=prompts
                )
            else:
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

    def plot_confusion_matrix(self, save_path=None):
        """Generate and save confusion matrix plot."""
        if len(self.all_predictions) == 0 or len(self.all_labels) == 0:
            print("No predictions available for confusion matrix")
            return
        
        unique_labels = sorted(list(set(self.all_labels + self.all_predictions)))
        class_names = [self.classidx_to_name.get(label, str(label)) for label in unique_labels]
        
        cm = confusion_matrix(self.all_labels, self.all_predictions, labels=unique_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.savefig(osp.join(self.run_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {osp.join(self.run_folder, 'confusion_matrix.png')}")
        plt.close()
        
        report = classification_report(self.all_labels, self.all_predictions, 
                                     labels=unique_labels, target_names=class_names)
        report_path = osp.join(self.run_folder, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

    def save_results_summary(self):
        """Save a comprehensive summary of results."""
        if len(self.all_predictions) == 0 or len(self.all_labels) == 0:
            print("No results to summarize")
            return
            
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Diffusion Classification Results Summary\n")
            f.write("=======================================\n\n")
            
            correct = sum(1 for p, l in zip(self.all_predictions, self.all_labels) if p == l)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Overall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct predictions: {correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Background removal: {self.args.remove_background}\n")
            f.write(f"Visualization enabled: {self.args.visualize}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Loss function: {self.args.loss}\n")
            f.write(f"Data type: {self.args.dtype}\n")
            f.write(f"Interpolation: {self.args.interpolation}\n")
            f.write(f"Adaptive sampling - n_samples: {self.args.n_samples}\n")
            f.write(f"Adaptive sampling - to_keep: {self.args.to_keep}\n\n")
            
            if hasattr(self, 'classidx_to_name') and len(self.classidx_to_name) > 0:
                f.write("Per-class Results:\n")
                class_stats = {}
                for true_label, pred_label in zip(self.all_labels, self.all_predictions):
                    if true_label not in class_stats:
                        class_stats[true_label] = {'total': 0, 'correct': 0}
                    class_stats[true_label]['total'] += 1
                    if true_label == pred_label:
                        class_stats[true_label]['correct'] += 1
                
                for class_idx, stats in sorted(class_stats.items()):
                    class_name = self.classidx_to_name.get(class_idx, str(class_idx))
                    class_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    f.write(f"{class_name}: {stats['correct']}/{stats['total']} ({class_acc:.1f}%)\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation(self):
        if self.args.subset_path is not None:
            idxs = np.load(self.args.subset_path).tolist()
        elif self.args.samples_per_class is not None:
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
                    self.all_predictions.append(data['pred'])
                    self.all_labels.append(data['label'])
                continue
            
            image, label = self.target_dataset[i]
            
            # Remove background if requested
            if self.args.remove_background:
                image = self.process_image_tensor(image)
            
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215

            text_embeddings_to_use = self.text_embeddings
            
            # Get prompts for this evaluation
            prompts = self.prompts_df.prompt.tolist()
                
            _, pred_idx, pred_errors = self.eval_prob_adaptive(
                self.unet, x0, text_embeddings_to_use, self.scheduler, 
                self.args, img_input, label, self.latent_size, self.all_noise,
                prompts=prompts
            )
            
            pred = self.prompts_df.classidx[pred_idx]
                
            torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
            if pred == label:
                correct += 1
            total += 1
            
            self.all_predictions.append(pred)
            self.all_labels.append(label)
        
        self.plot_confusion_matrix()
        self.save_results_summary()


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
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, default='prompts/pets_prompts.csv', help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--samples_per_class', type=int, default=1, help='Number of samples per class for balanced subset')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'), help='Type of loss to use')
    parser.add_argument('--remove_background', action='store_true', default=False, help='Remove background using BiRefNet before classification')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize error heatmaps during evaluation')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, default=[1])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[10])

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    evaluator = DiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()