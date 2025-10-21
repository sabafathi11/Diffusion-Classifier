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
import re
from diffusers.models.attention_processor import AttnProcessor

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

# Fruit color mapping (natural colors)
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



def parse_compound_unnatural_filename(filename):
    """
    Parse unnatural fruit combination filename.
    Expected format: compound_fruit1_color1_fruit2_color2_number.png
    Example: compound_banana_green_brinjal_red_220.png
    
    Returns: (fruit1, color1, fruit2, color2) or None if parsing fails
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Expected: ['compound', fruit1, color1, fruit2, color2, number]
    if len(parts) >= 6 and parts[0] == 'compound':
        fruit1 = parts[1]
        color1 = parts[2]
        fruit2 = parts[3]
        color2 = parts[4]
        return fruit1, color1, fruit2, color2
    
    return None


def parse_single_unnatural_filename(filename):
    """
    Parse single unnatural fruit filename.
    Expected format: fruit_color_number1_number2.png
    Example: banana_green_006_202011.png
    Returns: (fruit, color) or None if parsing fails
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Expected: [fruit, color, number1, number2]
    if len(parts) >= 3:
        fruit = parts[0]
        color = parts[1]
        
        # Verify color is valid
        x = random.choice(ALL_COLORS)
        if color in ALL_COLORS:
            return fruit, color
    
    return None

class SaveAttnProcessor(AttnProcessor):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Compute queries/keys/values
        batch_size, seq_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(context)
        value = attn.to_v(context)

        # Reshape to (batch, heads, seq, dim)
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Attention scores -> probs
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
        attn_probs  = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Save the attention map
        if encoder_hidden_states is not None:
            self.store.append(attn_probs.detach().cpu())

        # Apply attention to values
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Final projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



class CABDataset:
    """Custom dataset for CAB fruit images"""
    def __init__(self, root_dir, mode='compound', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'compound':
            self.image_paths = glob.glob(osp.join(root_dir, "compound", "*.jpg"))
        elif mode == 'single':
            self.image_paths = glob.glob(osp.join(root_dir, "single", "*.jpg"))
        elif mode == 'compound_unnatural':
            # Look for both jpg and png in unnatural_fruit_combinations
            self.image_paths = glob.glob(osp.join(root_dir, "compound_unnatural", "*.jpg"))
            self.image_paths += glob.glob(osp.join(root_dir, "compound_unnatural", "*.png"))
        elif mode == 'single_unnatural':
            # Look for both jpg and png in single_unnatural folder
            self.image_paths = glob.glob(osp.join(root_dir, "single_unnatural", "*.jpg"))
            self.image_paths += glob.glob(osp.join(root_dir, "single_unnatural", "*.png"))
        
        print(f"Found {len(self.image_paths)} {mode} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        filename = osp.basename(image_path)
        
        if self.mode == 'compound_unnatural':
            # Parse the filename to extract fruit names and colors
            parsed = parse_compound_unnatural_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
                colors = {}
            else:
                fruit1, color1, fruit2, color2 = parsed
                fruits = [fruit1, fruit2]
                colors = {fruit1: color1, fruit2: color2}
            return image, fruits, colors, image_path
        elif self.mode == 'single_unnatural':
            # Parse the filename to extract fruit name and color
            parsed = parse_single_unnatural_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
                color = None
            else:
                fruit, color = parsed
                fruits = [fruit]
            return image, fruits, color, image_path
        else:
            # Original behavior for compound and single modes
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
        if self.args.mode == 'single' or self.args.mode == 'single_unnatural':
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
            if self.args.mode in ['compound', 'compound_unnatural']:
                prompt = f"A {color.lower()} {target_fruit.lower()} and another object."
            else:
                prompt = f"A {color.lower()} {target_fruit.lower()}."
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

                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors
    
    def eval_error_visualize(self, unet, scheduler, vae, tokenizer, original_image, latent, all_noise, ts, noise_idxs,
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
                viz_data = defaultdict(lambda: {
                    'images': [], 
                    'errors': [], 
                    'timesteps': [],
                    'attention_maps': []  # Add attention storage
                })
            
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

                    attention_store = []
                    unet.set_attn_processor(SaveAttnProcessor(attention_store))

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
                    
                    if visualize:
                        for i in range(len(batch_ts)):
                            prompt_idx = batch_text_idxs[i]
                            viz_data[prompt_idx]['images'].append(reconstructed_image[i].detach().cpu())
                            viz_data[prompt_idx]['errors'].append(error[i].detach().cpu())
                            viz_data[prompt_idx]['timesteps'].append(batch_ts[i].item())
                            # Store attention for this sample
                            viz_data[prompt_idx]['attention_maps'].append([attn[i:i+1].detach().cpu() for attn in attention_store])
                    
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
                    
                    # Save error heatmap
                    save_path = osp.join(self.run_folder, f'error_heatmap_prompt{prompt_idx}_{timestamp}.png')
                    self.visualize_error_heatmap(
                        original_image, target_fruit, recon_images, error_maps, 
                        timesteps, prompt_text, save_path=save_path
                    )
                    
                    # Save attention visualization across all timesteps
                    p = prompt_text[:-1].lower() if prompt_text.endswith('.') else prompt_text.lower()
                    tokens_to_vis = p.split()[1:3]  # visualize color and fruit tokens
                    
                    attn_save_path = osp.join(self.run_folder, f'attention_prompt{prompt_idx}_{timestamp}.png')
                    self.visualize_token_attention_grid(
                        attention_maps=data['attention_maps'],
                        image=original_image,
                        prompt=prompt_text,
                        tokenizer=tokenizer,
                        tokens_to_vis=tokens_to_vis,
                        timesteps=timesteps,
                        save_path=attn_save_path
                    )
            
            return pred_errors


    def visualize_token_attention_grid(self, attention_maps, image, prompt, tokenizer, 
                                    tokens_to_vis, timesteps, upsample_size=(512, 512), 
                                    save_path=None):
        """
        Visualize attention maps across all timesteps in a grid layout.
        
        Args:
            attention_maps: List of attention stores, one per timestep. Each is a list of 
                        tensors [1, heads, H*W, tokens] from cross-attn layers
            image: torch.Tensor of shape [1, C, H, W] or [C, H, W] (original image)
            prompt: str, e.g. "A red apple"
            tokenizer: CLIP tokenizer used to encode text
            tokens_to_vis: list of words to visualize, e.g. ["red", "apple"]
            timesteps: tensor of timesteps corresponding to each attention map
            upsample_size: size to upsample attention map to, e.g. (512, 512)
            save_path: Path to save the figure
        """
        # Tokenize prompt
        tokens = tokenizer.tokenize(prompt)
        tokens = [t.replace("</w>", "") for t in tokens]
        
        # Find token indices for visualization
        token_indices = []
        for word in tokens_to_vis:
            matches = [i for i, tok in enumerate(tokens) if word in tok]
            if not matches:
                print(f"Token '{word}' not found in: {tokens}")
            else:
                token_indices.append((word, matches))
        
        if not token_indices:
            print("No tokens found to visualize")
            return
        
        # Sort by timestep
        sort_idx = torch.argsort(timesteps)
        timesteps = timesteps[sort_idx]
        attention_maps = [attention_maps[i] for i in sort_idx]
        
        # Prepare image
        image_to_plot = image[0] if image.ndim == 4 else image
        image_np = image_to_plot.permute(1, 2, 0).cpu().numpy()
        
        if image_np.dtype not in ['float32', 'float64']:
            image_np = image_np.astype('float32')
        if image_np.min() < 0:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Determine grid layout
        num_timesteps = len(attention_maps)
        num_tokens = len(token_indices)
        max_cols = 6
        num_rows = (num_timesteps + max_cols - 1) // max_cols
        num_cols = min(num_timesteps, max_cols)
        
        # Create figure for each token
        for token_word, token_word_ids in token_indices:
            fig = plt.figure(figsize=(5 * num_cols, 10 * num_rows))
            gs = fig.add_gridspec(num_rows + 1, num_cols, hspace=0.3, wspace=0.3)
            
            # Plot original image spanning first row
            ax_orig = fig.add_subplot(gs[0, :])
            ax_orig.imshow(image_np)
            ax_orig.set_title(f'Original Image\nPrompt: "{prompt}"\nToken: "{token_word}"', 
                            fontsize=12, fontweight='bold')
            ax_orig.axis('off')
            
            # Process and plot attention maps for each timestep
            for idx, (attn_store, t) in enumerate(zip(attention_maps, timesteps)):
                row = (idx // num_cols) + 1
                col = idx % num_cols
                
                # Process attention maps
                resized = []
                for attn in attn_store:
                    # Remove batch dimension if present
                    if attn.dim() == 4:
                        attn = attn.squeeze(0)  # [heads, H*W, tokens]
                    
                    heads, HW, tokens = attn.shape
                    h = w = int(HW ** 0.5)
                    attn_map = attn.view(heads, h, w, tokens).permute(0, 3, 1, 2)  # [heads, tokens, h, w]
                    attn_map = F.interpolate(attn_map, size=(64, 64), mode="bilinear", align_corners=False)
                    resized.append(attn_map)
                
                attn_all = torch.cat(resized, dim=0)  # combine heads+layers
                attn_mean = attn_all.mean(0)  # [tokens, 64, 64]
                
                # Average over all token pieces that match this word
                heatmap = attn_mean[token_word_ids].mean(0)  # [64, 64]
                
                # Normalize heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Upsample to image size
                heatmap_up = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                heatmap_up = F.interpolate(heatmap_up, size=upsample_size, mode="bilinear", align_corners=False)
                heatmap_up = heatmap_up.squeeze().cpu().numpy()
                
                # Plot overlay
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(image_np)
                im = ax.imshow(heatmap_up, cmap='jet', alpha=0.6, interpolation='nearest')
                ax.axis('off')
                ax.set_title(f't={t.item()}\nAttn: {heatmap.mean():.4f}', fontsize=10)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            
            plt.suptitle(f'Cross-Attention Maps for "{token_word}" Across All Timesteps', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            if save_path:
                # Create unique filename for each token
                base_path = save_path.replace('.png', '')
                token_save_path = f"{base_path}_{token_word}.png"
                plt.savefig(token_save_path, dpi=150, bbox_inches='tight')
                print(f"Attention visualization for '{token_word}' saved to {token_save_path}")
                plt.close()
        return fig
    
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
            
            if args.visualize:
                pred_errors = self.eval_error_visualize(
                    unet, scheduler, self.vae, self.tokenizer, image, latent, all_noise, ts, noise_idxs,
                    text_embeds, text_embed_idxs, target_fruit, args.batch_size, args.dtype, 
                    args.loss, T, visualize=True, prompts=prompts
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
            prompts=prompts
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[target_fruit.lower()]
        other_fruit = fruit2 if target_fruit == fruit1 else fruit1
        other_fruit_color = FRUIT_COLORS[other_fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, other_fruit_color)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx

    def perform_color_classification_compound_unnatural(self, image, fruit1, fruit2, target_fruit, target_color, other_fruit_color):
        """Perform color classification for a specific fruit in unnatural compound image"""
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
            prompts=prompts
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = target_color  # Use the color from filename
        
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
            self.args, img_input, fruit, self.latent_size, self.all_noise, prompts=prompts
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx

    def perform_color_classification_single_unnatural(self, image, fruit, target_color):
        """Perform color classification for a single unnatural fruit image"""
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
            # x0 = torch.randn_like(x0)

        all_losses, pred_idx, pred_errors = self.eval_prob_adaptive(
            self.unet, x0, text_embeddings, self.scheduler, 
            self.args, img_input, fruit, self.latent_size, self.all_noise, prompts=prompts
        )
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = target_color  # Use the color from filename
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx


    def save_confusion_matrix(self):
        """Save confusion matrix visualization and CSV for single mode"""
        if self.args.mode not in ['single', 'single_unnatural']:
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
        plt.title('Color Classification Confusion Matrix')
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
            f.write("==========================================================\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                other_fruit_acc = (self.other_fruit_predictions / self.total_classifications * 100)
                other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                
                f.write(f"Color Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.args.mode in ['compound', 'compound_unnatural']:
                    f.write(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
                    f.write(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                else:
                    f.write(f"Incorrect predictions: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            if self.args.mode in ['compound', 'compound_unnatural']:
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

    def run_evaluation_compound_unnatural(self):
        """Run evaluation for unnatural fruit combination images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = get_formatstr(len(self.target_dataset) * 2 - 1)
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.1f}%, Other Fruit: {other_fruit_acc:.1f}% ({self.total_classifications})')
            
            image, fruits, colors, image_path = self.target_dataset[i]
            
            if len(fruits) != 2:
                print(f"Skipping {image_path}: Expected 2 fruits, got {len(fruits)}")
                continue
            
            if not colors or len(colors) != 2:
                print(f"Skipping {image_path}: Could not parse colors from filename")
                continue
                
            fruit1, fruit2 = fruits
            
            # Verify colors are valid
            color1 = colors.get(fruit1)
            color2 = colors.get(fruit2)
            
            if color1 not in ALL_COLORS or color2 not in ALL_COLORS:
                print(f"Skipping {image_path}: Invalid colors {color1}, {color2}")
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
                
                # Get the target color and other fruit color from the filename
                target_color = colors[target_fruit]
                other_fruit = fruit2 if target_fruit == fruit1 else fruit1
                other_fruit_color = colors[other_fruit]
                
                predicted_color, correct_color, prediction_type, prompts, pred_idx = self.perform_color_classification_compound_unnatural(
                    image, fruit1, fruit2, target_fruit, target_color, other_fruit_color
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
                    target_color=target_color,
                    fruits=[fruit1, fruit2],
                    colors=colors,
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


    def run_evaluation_single_unnatural(self):
        """Run evaluation for single unnatural images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = get_formatstr(len(self.target_dataset) - 1)
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.2f}% ({self.correct_predictions}/{self.total_classifications})')
            
            image, fruits, color, image_path = self.target_dataset[i]
            
            if len(fruits) != 1:
                print(f"Skipping {image_path}: Expected 1 fruit, got {len(fruits)}")
                continue
            
            if color is None or color not in ALL_COLORS:
                print(f"Skipping {image_path}: Invalid or missing color in filename")
                continue
                
            fruit = fruits[0]
            
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
            
            predicted_color, correct_color, prediction_type, prompts, pred_idx = self.perform_color_classification_single_unnatural(
                image, fruit, color
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
                target_color=color,
                fruits=[fruit],
                prompts=prompts,
                image_path=image_path,
                classification_idx=i
            ), fname)


    def run_evaluation(self):
        """Run evaluation based on mode"""
        if self.args.mode == 'compound':
            self.run_evaluation_compound()
        elif self.args.mode == 'compound_unnatural':
            self.run_evaluation_compound_unnatural()
        elif self.args.mode == 'single_unnatural':
            self.run_evaluation_single_unnatural()
        else:
            self.run_evaluation_single()
        
        # Generate summary
        self.save_results_summary()
        
        # Save confusion matrix for single and single_unnatural modes
        if self.args.mode in ['single', 'single_unnatural']:
            self.save_confusion_matrix()
        
        # Print final results
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Color Classification Results ({self.args.mode.title()} Mode):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.args.mode in ['compound', 'compound_unnatural']:
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
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/color',
                        help='Path to color folder')
    parser.add_argument('--mode', type=str, default='compound_unnatural', 
                        choices=['compound', 'single', 'compound_unnatural', 'single_unnatural'],
                        help='Mode: compound for two-fruit images, single for single-fruit images, ' + 
                            'compound_unnatural for unnatural color combinations, single_unnatural for single unnatural fruits')
    
    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
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