import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
import torchvision.transforms as transforms
from datetime import datetime
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class SaveAttnProcessor(AttnProcessor):
    """Custom attention processor to capture cross-attention maps"""
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
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Attention scores -> probs
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Save the attention map (only cross-attention)
        if encoder_hidden_states is not None:
            self.store.append(attn_probs.detach().cpu())

        # Apply attention to values
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Final projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttentionVisualizer:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-base"):
        """Initialize the attention visualizer with a Stable Diffusion model"""
        print(f"Loading model: {model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Use DDIM scheduler for consistency
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        
        print("Model loaded successfully!")

    def preprocess_image(self, image_path, size=512):
        """Load and preprocess image (matching your classification code)"""
        image = Image.open(image_path).convert('RGB')
        
        # Pad to square (matching your pad_to_square function)
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        
        transform = transforms.Compose([
            transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=255),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return transform(image).unsqueeze(0)

    def get_text_embeddings(self, prompt):
        """Get text embeddings for a prompt"""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        
        return text_embeddings

    def encode_image(self, image_tensor):
        """Encode image to latent space"""
        with torch.no_grad():
            img_input = image_tensor.to(device)
            if device == "cuda":
                img_input = img_input.half()
            latent = self.vae.encode(img_input).latent_dist.mean
            latent = latent * 0.18215  # Matching your classification code
        return latent

    def capture_attention(self, latent, text_embeddings, timesteps=[999, 800, 600, 400, 200, 50]):
        """Capture attention maps at specified timesteps"""
        attention_maps = []
        
        for t in timesteps:
            # Generate noise
            noise = torch.randn_like(latent)
            t_tensor = torch.tensor([t], device=device)
            
            # Get alpha values from scheduler
            alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
            sqrt_alpha_prod = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1).to(device)
            sqrt_one_minus_alpha_prod = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1).to(device)
            
            # Add noise to latent
            noised_latent = latent * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
            
            # Setup attention capture
            attention_store = []
            self.unet.set_attn_processor(SaveAttnProcessor(attention_store))
            
            # Run UNet forward pass
            with torch.no_grad():
                t_input = t_tensor.half() if device == "cuda" else t_tensor
                text_input = text_embeddings.half() if device == "cuda" else text_embeddings
                noised_latent_input = noised_latent.half() if device == "cuda" else noised_latent
                
                _ = self.unet(
                    noised_latent_input,
                    t_input,
                    encoder_hidden_states=text_input
                ).sample
            
            attention_maps.append((t, attention_store))
        
        return attention_maps

    def visualize_attention(self, image_path, prompt, words_to_visualize=None, 
                          output_dir="attention_outputs", timesteps=None):
        """
        Main function to visualize attention maps
        
        Args:
            image_path: Path to input image
            prompt: Text prompt to analyze
            words_to_visualize: List of words to visualize (if None, extracts from prompt)
            output_dir: Directory to save outputs
            timesteps: List of timesteps to visualize (if None, uses default)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess
        print(f"Processing image: {image_path}")
        print(f"Prompt: {prompt}")
        image_tensor = self.preprocess_image(image_path)
        latent = self.encode_image(image_tensor)
        text_embeddings = self.get_text_embeddings(prompt)
        
        # Get tokens
        tokens = self.tokenizer.tokenize(prompt)
        tokens = [t.replace("</w>", "") for t in tokens]
        print(f"Tokens: {tokens}")
        
        # Determine words to visualize
        if words_to_visualize is None:
            # Extract content words from prompt
            skip_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            words = prompt.lower().replace(',', '').replace('.', '').split()
            words_to_visualize = [w for w in words if w not in skip_words]
        
        print(f"Visualizing words: {words_to_visualize}")
        
        # Capture attention
        if timesteps is None:
            timesteps = [999, 800, 600, 400, 200, 50]
        
        print("Capturing attention maps...")
        attention_data = self.capture_attention(latent, text_embeddings, timesteps)
        
        # Visualize for each word
        for word in words_to_visualize:
            self.plot_attention_grid(
                image_tensor, attention_data, tokens, word, 
                prompt, output_dir
            )
        
        print(f"Attention maps saved to {output_dir}")

    def plot_attention_grid(self, image_tensor, attention_data, tokens, 
                           target_word, prompt, output_dir):
        """Plot attention maps in a grid for a specific word"""
        # Find token indices for target word
        target_lower = target_word.lower()
        token_indices = [i for i, tok in enumerate(tokens) if target_lower in tok.lower() or tok.lower() in target_lower]
        
        if not token_indices:
            print(f"Warning: Word '{target_word}' not found in tokens: {tokens}")
            return
        
        print(f"Visualizing '{target_word}' using token indices {token_indices}: {[tokens[i] for i in token_indices]}")
        
        # Prepare image for display
        image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        image_np = np.clip(image_np, 0, 1)
        
        # Create grid layout
        num_timesteps = len(attention_data)
        ncols = min(3, num_timesteps)
        nrows = (num_timesteps + ncols - 1) // ncols + 1  # +1 for original image
        
        fig = plt.figure(figsize=(6 * ncols, 6 * nrows))
        
        # Plot original image
        ax = plt.subplot(nrows, ncols, 1)
        ax.imshow(image_np)
        ax.set_title(f'Original Image\n"{prompt}"', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Process and plot attention for each timestep
        for idx, (t, attention_store) in enumerate(attention_data):
            # Process attention maps (matching your classification code)
            resized = []
            for attn in attention_store:
                # Remove batch dimension if present
                if attn.dim() == 4:
                    attn = attn.squeeze(0)  # [heads, H*W, tokens]
                
                heads, HW, tokens_dim = attn.shape
                h = w = int(HW ** 0.5)
                attn_map = attn.view(heads, h, w, tokens_dim).permute(0, 3, 1, 2)  # [heads, tokens, h, w]
                attn_map = F.interpolate(attn_map, size=(64, 64), mode="bilinear", align_corners=False)
                resized.append(attn_map)
            
            attn_all = torch.cat(resized, dim=0)  # combine heads+layers
            attn_mean = attn_all.mean(0)  # [tokens, 64, 64]
            
            # Average over token indices for target word
            heatmap = attn_mean[token_indices].mean(0)  # [64, 64]
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Upsample to image size
            heatmap_up = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            heatmap_up = F.interpolate(heatmap_up, size=(512, 512), mode="bilinear", align_corners=False)
            heatmap_up = heatmap_up.squeeze().cpu().numpy()
            
            # Plot overlay
            ax = plt.subplot(nrows, ncols, idx + 2)
            ax.imshow(image_np)
            im = ax.imshow(heatmap_up, cmap='jet', alpha=0.6, interpolation='nearest')
            ax.axis('off')
            ax.set_title(f't={t}\nAttn: {heatmap.mean():.4f}', fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Cross-Attention Maps for "{target_word}"', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_{target_word.replace(' ', '_')}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Visualize cross-attention maps for images and prompts")
    parser.add_argument('--image', type=str, default='lab-coat.png', help='Path to input image')
    parser.add_argument('--prompt', type=str, default='a lab coat', help='Text prompt to analyze')
    parser.add_argument('--words', type=str, nargs='*', default=None, 
                       help='Specific words to visualize (default: all content words)')
    parser.add_argument('--output_dir', type=str, default='attention_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-2-base',
                       help='Stable Diffusion model to use')
    parser.add_argument('--timesteps', type=int, nargs='+', 
                       default=[999, 800, 600, 400, 200, 50],
                       help='Timesteps to visualize')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(model_id=args.model)
    
    # Run visualization
    visualizer.visualize_attention(
        image_path=args.image,
        prompt=args.prompt,
        words_to_visualize=args.words,
        output_dir=args.output_dir,
        timesteps=args.timesteps
    )
    
    print("\nDone! Check the output directory for results.")


if __name__ == '__main__':
    main()