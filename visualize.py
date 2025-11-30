import argparse
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import tqdm
from diffusion.models import get_sd_model, get_scheduler_config
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from collections import defaultdict
from PIL import Image
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from diffusers.models.attention_processor import AttnProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def pad_to_square(image):
    """Pad image to square shape"""
    width, height = image.size
    max_dim = max(width, height)
    
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    
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


class SaveAttnProcessor(AttnProcessor):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(context)
        value = attn.to_v(context)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)

        if encoder_hidden_states is not None:
            self.store.append(attn_probs.detach().cpu())

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class DiffusionHeatmapVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        self._setup_models()
        self._setup_output_dir()
        
    def _setup_models(self):
        """Initialize diffusion model components"""
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = get_sd_model(self.args)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        torch.backends.cudnn.benchmark = True
        print(f"Loaded Stable Diffusion v{self.args.version}")
        
    def _setup_output_dir(self):
        """Create output directory"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        print(f"Output directory: {self.args.output_dir}")
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image"""
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        
        image = Image.open(image_path)
        image_tensor = transform(image)
        
        return image_tensor
    
    def get_text_embedding(self, prompt):
        """Get text embedding for a prompt"""
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        return text_embedding
    
    def visualize_heatmap(self, image_path, prompt, timestep):
        """
        Visualize either error heatmap or attention map for a single timestep.
        
        Args:
            image_path: Path to the input image
            prompt: Text prompt to condition the denoising
            timestep: Single timestep to evaluate
        """
        print(f"\nProcessing image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Timestep: {timestep}")
        print(f"Mode: {self.args.mode}")
        
        # Load and encode image
        image_tensor = self.load_and_preprocess_image(image_path)
        
        with torch.no_grad():
            img_input = image_tensor.to(self.device).unsqueeze(0)
            if self.args.dtype == 'float16':
                img_input = img_input.half()
            latent = self.vae.encode(img_input).latent_dist.mean
            latent *= 0.18215
        
        # Get text embedding
        text_embedding = self.get_text_embedding(prompt)
        if self.args.dtype == 'float16':
            text_embedding = text_embedding.half()
        
        # Generate noise
        latent_size = self.args.img_size // 8
        noise = torch.randn((1, 4, latent_size, latent_size), device=self.device)
        if self.args.dtype == 'float16':
            noise = noise.half()
        
        # Add noise to latent
        t_tensor = torch.tensor([timestep], device=self.device)
        alpha_prod_t = self.scheduler.alphas_cumprod[t_tensor.cpu()].to(self.device)
        sqrt_alpha_prod = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1)
        
        if self.args.dtype == 'float16':
            sqrt_alpha_prod = sqrt_alpha_prod.half()
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.half()
        
        noised_latent = latent * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
        
        # Predict noise with attention tracking if needed
        attention_store = []
        if self.args.mode == 'attention':
            self.unet.set_attn_processor(SaveAttnProcessor(attention_store))
        
        with torch.no_grad():
            noise_pred = self.unet(noised_latent, t_tensor, encoder_hidden_states=text_embedding).sample
        
        # Create filename base
        image_name = osp.splitext(osp.basename(image_path))[0]
        prompt_clean = prompt.replace(' ', '_')[:60]  # Limit prompt length in filename
        
        if self.args.mode == 'error':
            # Denoise to get predicted clean latent
            denoised_latent = (noised_latent - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
            
            # Decode to image (keep in float16 if that's what we're using)
            reconstructed_image = self.vae.decode(denoised_latent / self.vae.config.scaling_factor).sample
            
            # Compute error in pixel space (convert to float32 for computation)
            if self.args.dtype == 'float16':
                reconstructed_image_float = reconstructed_image.float()
                img_input_float = img_input.float()
            else:
                reconstructed_image_float = reconstructed_image
                img_input_float = img_input
            
            if self.args.loss == 'l2':
                error = F.mse_loss(img_input_float, reconstructed_image_float, reduction='none').mean(dim=1)
            elif self.args.loss == 'l1':
                error = F.l1_loss(img_input_float, reconstructed_image_float, reduction='none').mean(dim=1)
            elif self.args.loss == 'huber':
                error = F.huber_loss(img_input_float, reconstructed_image_float, reduction='none').mean(dim=1)
            
            error_map = error[0].detach().cpu().numpy()
            
            # Save error heatmap
            save_path = osp.join(
                self.args.output_dir,
                f"error_{timestep}_{prompt_clean}.png"
            )
            self.save_error_heatmap(error_map, save_path)
            
        elif self.args.mode == 'attention':
            if not self.args.attention_word:
                raise ValueError("--attention_word must be specified when using attention mode")
            
            # Process attention maps
            save_path = osp.join(
                self.args.output_dir,
                f"attention_{timestep}_{prompt_clean}_{self.args.attention_word}.png"
            )
            self.save_attention_heatmap(
                attention_store, img_input, prompt, 
                self.args.attention_word, save_path
            )
        
        print(f"Visualization saved to {save_path}")
    
    def save_error_heatmap(self, error_map, save_path):
        """Save error heatmap as a clean image without any labels"""
        fig, ax = plt.subplots(figsize=(self.args.img_size/100, self.args.img_size/100), dpi=100)
        ax.imshow(error_map, cmap='hot', interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Error heatmap saved to {save_path}")
    
    def save_attention_heatmap(self, attention_store, image, prompt, 
                               target_word, save_path, upsample_size=(512, 512)):
        """Save attention heatmap as a clean image without any labels"""
        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt)
        tokens = [t.replace("</w>", "") for t in tokens]
        
        # Find token indices for target word
        token_indices = [i for i, tok in enumerate(tokens) if target_word.lower() in tok.lower()]
        
        if not token_indices:
            print(f"Warning: Token '{target_word}' not found in prompt tokens: {tokens}")
            # Try to find partial matches
            token_indices = [i for i, tok in enumerate(tokens) if target_word.lower()[:3] in tok.lower()]
            if not token_indices:
                raise ValueError(f"Cannot find token '{target_word}' in prompt")
        
        print(f"Found token '{target_word}' at indices: {token_indices}")
        
        # Process attention maps
        resized = []
        for attn in attention_store:
            if attn.dim() == 4:
                attn = attn.squeeze(0)
            
            heads, HW, tokens = attn.shape
            h = w = int(HW ** 0.5)
            attn_map = attn.view(heads, h, w, tokens).permute(0, 3, 1, 2)
            attn_map = F.interpolate(attn_map, size=(64, 64), mode="bilinear", align_corners=False)
            resized.append(attn_map)
        
        attn_all = torch.cat(resized, dim=0)
        attn_mean = attn_all.mean(0)
        
        # Average over token pieces
        heatmap = attn_mean[token_indices].mean(0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Upsample to match image size
        heatmap_up = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap_up = F.interpolate(heatmap_up, size=upsample_size, mode="bilinear", align_corners=False)
        heatmap_up = heatmap_up.squeeze().cpu().numpy()
        
        # Save clean heatmap overlay
        fig, ax = plt.subplots(figsize=(self.args.img_size/100, self.args.img_size/100), dpi=100)
        
        # Prepare base image
        image_to_plot = image[0] if image.ndim == 4 else image
        image_np = image_to_plot.permute(1, 2, 0).cpu().numpy()
        
        if image_np.dtype not in ['float32', 'float64']:
            image_np = image_np.astype('float32')
        if image_np.min() < 0:
            image_np = (image_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        
        ax.imshow(image_np)
        ax.imshow(heatmap_up, cmap='jet', alpha=0.6, interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Attention heatmap saved to {save_path}")
    
    def process_directory(self):
        """Process all images in the input directory"""
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(osp.join(self.args.image_dir, ext)))
            image_paths.extend(glob.glob(osp.join(self.args.image_dir, ext.upper())))
        
        if not image_paths:
            print(f"No images found in {self.args.image_dir}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for image_path in image_paths:
            try:
                self.visualize_heatmap(image_path, self.args.prompt, self.args.timestep)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue


def main():
    parser = argparse.ArgumentParser(description='Visualize diffusion error or attention heatmap')
    
    # Input/Output
    parser.add_argument('--image_dir', type=str, default='/work/gn21/h62001/Diffusion-Classifier/samples',
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--prompt', type=str, default='a photo of a ostrich',
                       help='Text prompt for conditioning the denoising')
    
    # Visualization mode
    parser.add_argument('--mode', type=str, required=True, choices=['error', 'attention'],
                       help='Type of heatmap to generate: error or attention')
    parser.add_argument('--timestep', type=int, default=300,
                       help='Single timestep to evaluate (0-999)')
    parser.add_argument('--attention_word', type=str, default='ostrich',
                       help='Word to visualize attention for (required when mode=attention)')
    
    # Model args
    parser.add_argument('--version', type=str, default='2-0',
                       help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512),
                       help='Image size')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                       help='Model data type')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                       choices=('bilinear', 'bicubic', 'lanczos'),
                       help='Resize interpolation type')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'),
                       help='Loss function for error computation (only for error mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.timestep < 1000:
        raise ValueError("Timestep must be in range [0, 999]")
    
    if args.mode == 'attention' and not args.attention_word:
        raise ValueError("--attention_word must be specified when using attention mode")
    
    # Create visualizer and run
    visualizer = DiffusionHeatmapVisualizer(args)
    visualizer.process_directory()
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()