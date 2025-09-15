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

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)             # sets seed for CPU
torch.cuda.manual_seed(seed)        # sets seed for current GPU
torch.cuda.manual_seed_all(seed)    # sets seed for all GPUs (if you use multi-GPU)
np.random.seed(seed) 


class PromptLearner(torch.nn.Module):
    """
    Learn a single GLOBAL prompt (n_ctx vectors) and insert them into the
    text encoder input embeddings. Returns text encoder last_hidden_state
    with shape [n_prompts, seq_len, hidden_dim] — compatible with UNet's
    `encoder_hidden_states`.
    """
    def __init__(self, tokenizer, text_encoder, n_ctx=8, ctx_init=None, device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.n_ctx = n_ctx
        self.device = device if device is not None else next(text_encoder.parameters()).device
        self.hidden_size = text_encoder.config.hidden_size

        # learnable ctx vectors (initialized randomly or from ctx_init)
        if ctx_init is None:
            ctx_init = torch.randn(n_ctx, self.hidden_size) * 0.02
        self.ctx = torch.nn.Parameter(ctx_init.to(self.device))  # shape [n_ctx, D]

        # keep text encoder frozen (we want to update only ctx)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def get_prompt_embeds(self, classnames, max_length=None):
        enc = self.tokenizer(
            classnames,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)            # [B, L]
        attention_mask = enc["attention_mask"].to(self.device)  # [B, L]

        token_embeds = self.text_encoder.get_input_embeddings()(input_ids)  # [B, L, D]
        B, L, D = token_embeds.shape
        assert D == self.hidden_size

        if self.n_ctx >= L:
            raise ValueError("n_ctx must be < sequence length")

        prefix = token_embeds[:, :1, :]                                # BOS
        suffix = token_embeds[:, 1 + self.n_ctx:, :]                  # FIXED: start from 1 + n_ctx
        ctx_expand = self.ctx.unsqueeze(0).expand(B, -1, -1)           # [B, n_ctx, D]

        # Ensure all tensors have the same dtype
        ctx_expand = ctx_expand.to(token_embeds.dtype)
        
        new_inputs_embeds = torch.cat([prefix, ctx_expand, suffix], dim=1)
        
        # FIXED: Handle padding properly
        if new_inputs_embeds.shape[1] < L:
            padding_size = L - new_inputs_embeds.shape[1]
            padding = torch.zeros(B, padding_size, D, device=self.device, dtype=new_inputs_embeds.dtype)
            new_inputs_embeds = torch.cat([new_inputs_embeds, padding], dim=1)
        new_inputs_embeds = new_inputs_embeds[:, :L, :]                # ensure length = 77

        # FIXED: Handle attention mask properly and ensure correct dtype
        prefix_mask = attention_mask[:, :1]
        ctx_mask = torch.ones((B, self.n_ctx), device=self.device, dtype=attention_mask.dtype)
        suffix_mask = attention_mask[:, 1 + self.n_ctx:]
        
        new_attention_mask = torch.cat([prefix_mask, ctx_mask, suffix_mask], dim=1)
        
        if new_attention_mask.shape[1] < L:
            padding_size = L - new_attention_mask.shape[1]
            padding = torch.zeros((B, padding_size), device=self.device, dtype=attention_mask.dtype)
            new_attention_mask = torch.cat([new_attention_mask, padding], dim=1)
        new_attention_mask = new_attention_mask[:, :L]                 # ensure length = 77
        
        # FIXED: Don't convert attention mask dtype - keep it as integers for proper processing
        # The attention mask should remain as integers (0s and 1s)
        
        # Manually pass through the text model components
        # Get position embeddings
        seq_length = new_inputs_embeds.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        
        # Combine token embeddings with position embeddings
        embeddings = new_inputs_embeds + position_embeds
        
        # FIXED: Create extended attention mask for encoder
        # Convert 2D attention mask to 4D for multi-head attention
        # Shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        extended_attention_mask = new_attention_mask[:, None, None, :]
        
        # Convert to float and apply masking values
        # 1.0 for tokens that should be attended to, large negative for padding
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through the encoder with properly formatted attention mask
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=embeddings,
            attention_mask=extended_attention_mask
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # Apply final layer norm
        hidden_states = self.text_encoder.text_model.final_layer_norm(hidden_states)
        
        return hidden_states
    
    def save(self, path: str):
        """Save the learned context vectors (and any other parameters)."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        """Load the learned context vectors."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)

def prompt_training_step(prompt_learner,
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
    prompt_learner.train()
    unet.eval()   # unet frozen in eval mode (no dropout etc). ensure params require_grad=False
    for p in unet.parameters():
        p.requires_grad = False

    B = latents.shape[0]
    C = len(classnames)

    # 1) get class prompt embeddings: [C, L, D]
    prompt_embeds = prompt_learner.get_prompt_embeds(classnames)  # on device

    # 2) sample noise and timesteps per image
    T = scheduler.config.num_train_timesteps if hasattr(scheduler.config, "num_train_timesteps") else len(scheduler.alphas_cumprod)
    timesteps = torch.randint(0, T, (B,), device=device, dtype=torch.long)  # one t per image
    noise = torch.randn_like(latents, device=device)  # [B, 4, H, W]

    # 3) compute noised latents (vectorized)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # ensure on device
    a = alphas_cumprod[timesteps].view(B, 1, 1, 1).sqrt()
    one_minus_a = (1 - alphas_cumprod[timesteps]).view(B, 1, 1, 1).sqrt()
    noised = latents * a + noise * one_minus_a  # [B, 4, H, W]

    # FIXED: Memory-optimized approach - process classes one by one
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

def run(args):

    transform = get_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = datasets.CIFAR10(root=DATASET_ROOT, train=(args.split == 'train'), download=True, transform=transform)
    classnames = dataset.classes
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

    prompt_learner = PromptLearner(tokenizer, text_encoder, n_ctx=8, device=device).to(device)
    optimizer = torch.optim.AdamW([prompt_learner.ctx], lr=1e-3)

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        for batch_idx, (images_batch, labels_batch) in enumerate(train_loader):
            if args.dtype == 'float16':
                images_batch = images_batch.to(device).half()
            else:
                images_batch = images_batch.to(device).float()
                
            with torch.no_grad():
                latents_batch = vae.encode(images_batch).latent_dist.mean
                latents_batch *= 0.18215

            loss_val = prompt_training_step(
                prompt_learner, unet, scheduler,
                latents_batch, labels_batch,
                classnames, optimizer, device
            )
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, train loss: {loss_val}")
    
    save_path = "templates/prompt_learner1.pt"
    prompt_learner.save(save_path)
    print(f"Saved learned prompt to {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                            help='Model data type to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()