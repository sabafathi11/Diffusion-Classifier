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

device = "cuda" if torch.cuda.is_available() else "cpu"

class WorkingPromptLearner(torch.nn.Module):
    """
    FIXED: Learn to map class embeddings to text embedding space properly
    """
    def __init__(self, tokenizer, text_encoder, classnames, device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.device = device if device is not None else next(text_encoder.parameters()).device
        self.hidden_size = text_encoder.config.hidden_size

        # FIXED: Create learnable vectors that will replace the class name tokens
        # Initialize them close to existing class name embeddings
        self.learned_embeddings = torch.nn.Parameter(
            torch.randn(self.num_classes, self.hidden_size, device=self.device) * 0.02
        )
        
        # Get the base template embeddings and find where to insert class-specific info
        template = "a photo of a {}"
        self.template_texts = [template.format("object") for _ in classnames]  # placeholder
        self.class_texts = [template.format(name) for name in classnames]
        
        # Encode the templates to get positions
        with torch.no_grad():
            # Get template with placeholder
            enc_template = tokenizer(
                self.template_texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            # Get actual class texts to see the difference
            enc_classes = tokenizer(
                self.class_texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            self.template_ids = enc_template["input_ids"].to(device)
            self.class_ids = enc_classes["input_ids"].to(device)
            self.attention_mask = enc_classes["attention_mask"].to(device)
            
            # Find positions where class names appear (differ from template)
            self.class_positions = []
            for i in range(self.num_classes):
                diff_mask = (self.template_ids[i] != self.class_ids[i])
                positions = torch.where(diff_mask)[0]
                # Usually the class name is one token, take the first differing position
                self.class_positions.append(positions[0].item() if len(positions) > 0 else 4)
                
            print(f"Class positions: {self.class_positions}")
            
            # Get base embeddings
            self.base_embeddings = text_encoder.get_input_embeddings()(self.class_ids)
            
            # Initialize learned embeddings close to actual class embeddings
            for i in range(self.num_classes):
                pos = self.class_positions[i]
                self.learned_embeddings.data[i] = self.base_embeddings[i, pos].clone() + torch.randn_like(self.base_embeddings[i, pos]) * 0.01

        # Freeze text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        print(f"Initialized {self.num_classes} learnable embeddings")

    def get_prompt_embeds(self):
        """
        Create embeddings by replacing class name positions with learned embeddings
        """
        # Start with base embeddings
        prompt_embeds = self.base_embeddings.clone()
        
        # Replace class-specific positions with learned embeddings
        for i in range(self.num_classes):
            pos = self.class_positions[i]
            prompt_embeds[i, pos] = self.learned_embeddings[i]
        
        # Pass through text encoder
        # Create position embeddings
        batch_size, seq_len = prompt_embeds.shape[0], prompt_embeds.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        
        # Combine token + position embeddings
        embeddings = prompt_embeds + position_embeds
        
        # Create attention mask for encoder (convert to 4D)
        extended_attention_mask = self.attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through encoder
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=embeddings,
            attention_mask=extended_attention_mask
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        hidden_states = self.text_encoder.text_model.final_layer_norm(hidden_states)
        
        return hidden_states

def working_training_step(prompt_learner,
                         vae,
                         unet,
                         scheduler,
                         images,
                         labels,
                         optimizer,
                         device,
                         temperature=0.07):  # Lower temperature for sharper distributions
    """
    Fixed training step with proper scaling
    """
    prompt_learner.train()
    unet.eval()
    vae.eval()
    
    # Ensure frozen models don't compute gradients
    for p in unet.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False

    B = images.shape[0]
    C = prompt_learner.num_classes

    # Encode images to latents
    with torch.no_grad():  # VAE doesn't need gradients for prompt learning
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Get prompt embeddings for all classes
    prompt_embeds = prompt_learner.get_prompt_embeds()  # [C, L, D]

    # Sample noise and timestep
    T = scheduler.config.num_train_timesteps
    timesteps = torch.randint(0, T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(latents, device=device)

    # Add noise to latents
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    a = alphas_cumprod[timesteps].view(B, 1, 1, 1).sqrt()
    one_minus_a = (1 - alphas_cumprod[timesteps]).view(B, 1, 1, 1).sqrt()
    noised = latents * a + noise * one_minus_a

    # Calculate reconstruction losses for each class
    losses = torch.zeros(B, C, device=device)
    
    for c in range(C):
        # Get prompt for class c
        pe_c = prompt_embeds[c:c+1].expand(B, -1, -1)  # [B, L, D]
        
        # Forward through UNet
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(images.dtype == torch.float16)):
            out = unet(noised, timesteps, encoder_hidden_states=pe_c)
        noise_pred = out.sample
        
        # Compute MSE loss per sample
        loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2, 3))
        losses[:, c] = loss_per_sample

    # Convert to logits with temperature scaling
    # Lower loss = better reconstruction = higher probability
    logits = -losses / temperature
    
    # Compute classification loss
    ce_loss = F.cross_entropy(logits, labels.to(device))
    
    # Backward pass
    optimizer.zero_grad()
    ce_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Calculate accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels.to(device)).float().mean().item()
    
    return ce_loss.item(), accuracy, logits.detach()

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

def run_working(args):
    print("=== WORKING PROMPT LEARNING ===")
    
    transform = get_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = datasets.CIFAR10(root=DATASET_ROOT, train=(args.split == 'train'), download=True, transform=transform)
    classnames = dataset.classes
    print(f"Classes: {classnames}")
    
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    if args.dtype == 'float16':
        text_encoder = text_encoder.half()
        unet = unet.half()
        vae = vae.half()

    # Create working prompt learner
    prompt_learner = WorkingPromptLearner(tokenizer, text_encoder, classnames, device=device).to(device)
    
    # Use a higher learning rate since we're learning meaningful representations
    optimizer = torch.optim.AdamW([prompt_learner.learned_embeddings], lr=1e-2, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10*len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )

    print("\n=== Starting Working Training ===")
    
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}/10")
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_idx, (images_batch, labels_batch) in enumerate(train_loader):
            
            if args.dtype == 'float16':
                images_batch = images_batch.to(device).half()
            else:
                images_batch = images_batch.to(device).float()

            loss_val, accuracy, logits = working_training_step(
                prompt_learner, vae, unet, scheduler,
                images_batch, labels_batch,
                optimizer, device
            )
            
            scheduler_lr.step()  # Update LR every step
            
            epoch_losses.append(loss_val)
            epoch_accuracies.append(accuracy)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss_val:.6f}, Accuracy: {accuracy:.4f}")
                print(f"Logits: {logits[0][:5]}")  # Show first 5 logits
                
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies) 
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} Summary - Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.4f}, LR: {current_lr:.2e}")
    
    # Save the model
    save_path = "templates/working_prompt_learner.pt"
    torch.save(prompt_learner.state_dict(), save_path)
    print(f"Saved working prompt learner to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--dtype', type=str, default='float32', choices=('float16', 'float32'),
                            help='Model data type to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    args = parser.parse_args()
    run_working(args)

if __name__ == "__main__":
    main()