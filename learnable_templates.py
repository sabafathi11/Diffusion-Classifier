import argparse
import numpy as np
import torch
import torch.nn.functional as F
from diffusion.models import get_sd_model
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets
from diffusion.utils import DATASET_ROOT
from torch.utils.data import DataLoader
from diffusion.datasets import get_target_dataset
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)             # sets seed for CPU
torch.cuda.manual_seed(seed)        # sets seed for current GPU
torch.cuda.manual_seed_all(seed)    # sets seed for all GPUs (if you use multi-GPU)
np.random.seed(seed) 


class TemplatePromptLearner(torch.nn.Module):
    """
    Learn a GLOBAL template embedding that gets concatenated with class name embeddings.
    The template is optimized during training and can be reused across different classes.
    """
    def __init__(self, tokenizer, text_encoder, n_template_tokens, class_max_length, template_init=None, device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.n_template_tokens = n_template_tokens
        self.class_max_length = class_max_length
        self.device = device if device is not None else next(text_encoder.parameters()).device
        self.hidden_size = text_encoder.config.hidden_size
        
        # Ensure total length doesn't exceed CLIP limit
        assert n_template_tokens + class_max_length <= 77, f"Total length {n_template_tokens + class_max_length} exceeds CLIP limit of 77"

        # learnable global template vectors (initialized randomly or from template_init)
        if template_init is None:
            template_init = torch.randn(n_template_tokens, self.hidden_size) * 0.02
        self.global_template = torch.nn.Parameter(template_init.to(self.device))  # shape [n_template_tokens, D]

        # keep text encoder frozen (we want to update only the global template)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def get_prompt_embeds(self, classnames, max_length=None):
        """
        Create embeddings by concatenating global template + class name embeddings
        Uses fixed lengths: n_template_tokens + class_max_length = 77 exactly
        """
        # Encode class names with controlled length
        enc = self.tokenizer(
            classnames,
            padding="max_length",
            truncation=True,
            max_length=self.class_max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)            # [B, class_max_length]
        attention_mask = enc["attention_mask"].to(self.device)  # [B, class_max_length]

        # Get class name embeddings through the text encoder
        with torch.no_grad():
            class_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            class_embeds = class_outputs.last_hidden_state  # [B, class_max_length, D]
        
        B, L, D = class_embeds.shape
        assert L == self.class_max_length
        assert D == self.hidden_size

        # Expand global template for batch
        template_expand = self.global_template.unsqueeze(0).expand(B, -1, -1)  # [B, n_template_tokens, D]
        template_expand = template_expand.to(class_embeds.dtype)
        
        # Concatenate global template + class embeddings
        # This will be exactly n_template_tokens + class_max_length = 77 tokens
        combined_embeds = torch.cat([template_expand, class_embeds], dim=1)  # [B, 77, D]
        
        # Create attention mask for combined embeddings
        template_mask = torch.ones((B, self.n_template_tokens), device=self.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([template_mask, attention_mask], dim=1)  # [B, 77]

        # Process through text encoder transformer layers manually to get final representations
        # We need to bypass the embedding layer since we already have embeddings
        
        # Get position embeddings for full sequence
        seq_length = combined_embeds.shape[1]  # Should be 77
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        
        # Add position embeddings
        embeddings = combined_embeds + position_embeds
        
        # Create extended attention mask for encoder
        extended_attention_mask = combined_attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through the encoder
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=embeddings,
            attention_mask=extended_attention_mask
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # Apply final layer norm
        hidden_states = self.text_encoder.text_model.final_layer_norm(hidden_states)
        
        return hidden_states
    
    def get_global_template(self):
        """Return the learned global template for inspection or reuse"""
        return self.global_template.detach()
    
    def save(self, path: str):
        """Save the learned global template."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        """Load the learned global template."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)

def prompt_training_step(template_learner,
                         unet,
                         scheduler,
                         latents,      # [B, 4, H, W] (already on device)
                         labels,       # [B] (long tensor, values in [0..C-1])
                         classnames,   # list[str] length C
                         optimizer,
                         device,
                         loss_kind='l2'):
    """
    MEMORY-OPTIMIZED version:
    Instead of expanding batch to B*C, process each class separately to save memory
    """
    template_learner.train()
    unet.eval()   # unet frozen in eval mode (no dropout etc). ensure params require_grad=False
    for p in unet.parameters():
        p.requires_grad = False

    B = latents.shape[0]
    C = len(classnames)

    # 1) get class prompt embeddings with global template: [C, L, D]
    prompt_embeds = template_learner.get_prompt_embeds(classnames)  # on device

    # 2) sample noise and timesteps per image
    T = scheduler.config.num_train_timesteps if hasattr(scheduler.config, "num_train_timesteps") else len(scheduler.alphas_cumprod)
    timesteps = torch.randint(0, T, (B,), device=device, dtype=torch.long)  # one t per image
    noise = torch.randn_like(latents, device=device)  # [B, 4, H, W]

    # 3) compute noised latents (vectorized)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # ensure on device
    a = alphas_cumprod[timesteps].view(B, 1, 1, 1).sqrt()
    one_minus_a = (1 - alphas_cumprod[timesteps]).view(B, 1, 1, 1).sqrt()
    noised = latents * a + noise * one_minus_a  # [B, 4, H, W]

    # Memory-optimized approach - process classes one by one
    losses_per_pair = torch.zeros(B, C, device=device)
    
    for c in range(C):
        # Get prompt embedding for this class
        pe_c = prompt_embeds[c:c+1].expand(B, -1, -1)  # [B, L, D]
        
        # Forward through UNet for this class
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(next(unet.parameters()).dtype == torch.float16)):
            out = unet(noised, timesteps, encoder_hidden_states=pe_c)
        noise_pred = out.sample  # [B, 4, H, W]
        
        # Compute loss for this class
        if loss_kind == 'l2':
            errs = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2, 3))  # [B]
        elif loss_kind == 'l1':
            errs = F.l1_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2, 3))
        else:
            raise NotImplementedError
            
        losses_per_pair[:, c] = errs

    # 4) logits: lower loss -> higher score, so use negative loss as logits
    logits = -losses_per_pair  # [B, C]

    # 5) classification loss
    ce_loss = F.cross_entropy(logits, labels.to(device))

    # 6) step
    optimizer.zero_grad()
    ce_loss.backward()
    optimizer.step()

    return ce_loss.item()

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

def save_checkpoint(epoch, template_learner, optimizer, loss, checkpoint_dir):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'template_learner_state_dict': template_learner.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")

def load_checkpoint(checkpoint_path, template_learner, optimizer):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    template_learner.load_state_dict(checkpoint['template_learner_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, resuming from epoch {start_epoch}")
    return start_epoch, loss

def run(args):

    transform = get_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = datasets.CIFAR10(root=DATASET_ROOT, train=(args.split == 'train'), download=True, transform=transform)
    classnames = dataset.classes
    print(classnames)
    #dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)

    #dtype_map = {'float16': torch.float16, 'float32': torch.float32}
    #torch_dtype = dtype_map[args.dtype]    
    
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    if args.dtype == 'float16':
        text_encoder = text_encoder.half()
        unet = unet.half()
        vae = vae.half()

    # Use the new TemplatePromptLearner with controlled lengths
    # Using 32 template tokens + 45 class tokens = 77 total (fits perfectly in CLIP)
    template_learner = TemplatePromptLearner(
        tokenizer, 
        text_encoder, 
        n_template_tokens=32, 
        class_max_length=45, 
        device=device
    ).to(device)
    
    # Optimizer only updates the global template
    optimizer = torch.optim.AdamW([template_learner.global_template], lr=1e-3)

    # Check for existing checkpoints and resume if requested
    checkpoint_dir = os.path.join(DATASET_ROOT, 'checkpoints')
    start_epoch = 0
    if args.resume:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if checkpoint_files:
            # Find the latest checkpoint
            epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
            latest_epoch = max(epochs)
            latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
            start_epoch, _ = load_checkpoint(latest_checkpoint, template_learner, optimizer)

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )

    for epoch in range(start_epoch, 10):
        print(f"Epoch {epoch + 1}/10")
        epoch_losses = []
        
        for batch_idx, (images_batch, labels_batch) in enumerate(train_loader):
            if args.dtype == 'float16':
                images_batch = images_batch.to(device).half()
            else:
                images_batch = images_batch.to(device).float()
                
            with torch.no_grad():
                latents_batch = vae.encode(images_batch).latent_dist.mean
                latents_batch *= 0.18215

            loss_val = prompt_training_step(
                template_learner, unet, scheduler,
                latents_batch, labels_batch,
                classnames, optimizer, device
            )
            
            epoch_losses.append(loss_val)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, train loss: {loss_val}")
        
        # Calculate average epoch loss
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Save checkpoint after each epoch
        save_checkpoint(epoch, template_learner, optimizer, avg_epoch_loss, checkpoint_dir)
        
        # Print template stats for monitoring
        if epoch % 2 == 0:
            template = template_learner.get_global_template()
            print(f"Template norm: {template.norm().item():.4f}, mean: {template.mean().item():.4f}")
    
    save_path = "templates/global_template_learner.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    template_learner.save(save_path)
    print(f"Saved learned global template to {save_path}")

    # Optionally save just the template embedding for reuse
    template_embedding_path = "templates/global_template_embedding.pt"
    torch.save(template_learner.get_global_template(), template_embedding_path)
    print(f"Saved global template embedding to {template_embedding_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                            help='Model data type to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()