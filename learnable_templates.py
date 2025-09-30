import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import DATASET_ROOT
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

"""
batch_size = 32-128
epoch_size = 50-200
n_ctx = 16
ctx_init = "a blurry photo of a"
timestep = 50
"""


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

class PromptLearner(nn.Module):
    def __init__(self, class_names, tokenizer, text_encoder, n_ctx=16, ctx_init=None, class_token_position="end", dtype=torch.float16):
        super().__init__()
        
        self.n_cls = len(class_names)
        self.class_names = class_names
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.class_token_position = class_token_position
        
        # Get text encoder embedding dimension from the actual model
        self.ctx_dim = self.text_encoder.get_input_embeddings().embedding_dim
        

        if ctx_init:
            # use given words to initialize context vectors
            n_ctx = len(ctx_init.split(" "))
            prompt_tokens = self.tokenizer(ctx_init, add_special_tokens=False, return_tensors="pt")
            with torch.no_grad():
                input_ids = prompt_tokens.input_ids.to(self.text_encoder.device)
                init_embeddings = self.text_encoder.get_input_embeddings()(input_ids).type(dtype)

            ctx_vectors = init_embeddings[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        name_lens = [self.tokenizer(name, return_tensors="pt").input_ids.shape[1] for name in self.class_names]
        prompts = [prompt_prefix + " " + name + "." for name in self.class_names]

        tokenized_prompts = torch.cat([self.tokenizer(p, return_tensors="pt").input_ids for p in prompts]).to(self.text_encoder.device)
        with torch.no_grad():
            embedding = self.text_encoder.get_input_embeddings()(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        position_ids = torch.arange(prompts.size(1), device=prompts.device).unsqueeze(0)
        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        hidden_states = prompts + position_embeddings
        mask = torch.ones(prompts.size(0), 1, prompts.size(1), prompts.size(1), device=prompts.device).type(hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            hidden_states,
            attention_mask=mask,
            output_hidden_states=False,
        )

        last_hidden_state = self.text_encoder.text_model.final_layer_norm(encoder_outputs[0])

        return last_hidden_state



def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    """
    Evaluate diffusion prediction error.
    """
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    
    # Store errors in a list when training to avoid in-place operations
    pred_errors_list = []
    
    idx = 0
    

    for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
        batch_ts = torch.tensor(ts[idx: idx + batch_size])
        noise = all_noise[noise_idxs[idx: idx + batch_size]]
        noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                        noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
        t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
        text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
        noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
        
        if loss == 'l2':
            error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
        elif loss == 'l1':
            error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
        elif loss == 'huber':
            error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
        else:
            raise NotImplementedError
        
        pred_errors_list.append(error)

        idx += len(batch_ts)

    pred_errors = torch.cat(pred_errors_list, dim=0)
    
    return pred_errors


def eval_prob_adaptive_differentiable(unet, latent, text_embeds, scheduler, args, 
                                    latent_size=64, all_noise=None):
    """
    Differentiable version of eval_prob_adaptive for training.
    Returns class probabilities based on negative diffusion errors.
    """
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    n_classes = len(text_embeds)
    
    if all_noise is None:
        all_noise = torch.randn((args.n_trials, 4, latent_size, latent_size), device=latent.device)
    
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
    
    n_timesteps = 50
    timesteps = torch.randint(0, T, (n_timesteps,), device=latent.device)

    class_errors = []
    
    for class_idx in range(n_classes):
        ts = timesteps.repeat(len(all_noise)).tolist()
        noise_idxs = list(range(len(all_noise))) * len(timesteps)
        text_embed_idxs = [class_idx] * len(ts)
        
        errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                          text_embeds, text_embed_idxs, args.batch_size, args.dtype, 
                          args.loss)
        
        # Average error for this class
        mean_error = errors.mean()
        class_errors.append(mean_error)
    
    # Stack errors and convert to probabilities (lower error = higher probability)
    class_errors = torch.stack(class_errors)  # Shape: [n_classes]
    
    # Use softmax with temperature to convert errors to probabilities
    # Negative because lower error should mean higher probability
    temperature = 0.1
    probs = F.softmax(-class_errors / temperature, dim=0)
    return probs, class_errors


def save_epoch_checkpoints(prompt_learner, epoch, save_dir, args):
    """Save learned embeddings and model state dict for the current epoch"""
    
    # Create epoch-specific directory
    epoch_dir = osp.join(save_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save prompt learner state dict
    prompt_state_path = osp.join(epoch_dir, 'prompt_learner.pth')
    torch.save(prompt_learner.state_dict(), prompt_state_path)
    
    # Save the learned embeddings (output of forward pass)
    with torch.no_grad():
        learned_embeddings = prompt_learner()
        embeddings_path = osp.join(epoch_dir, 'learned_embeddings.pth')
        torch.save(learned_embeddings.cpu(), embeddings_path)
    
    # Save the raw context vectors (learnable parameters)
    ctx_vectors_path = osp.join(epoch_dir, 'context_vectors.pth')
    torch.save(prompt_learner.ctx.data.cpu(), ctx_vectors_path)
    
    # Save metadata about the model
    metadata = {
        'epoch': epoch,
        'n_ctx': args.n_ctx,
        'ctx_dim': prompt_learner.ctx_dim,
        'n_classes': prompt_learner.n_cls,
        'class_names': prompt_learner.class_names,
        'class_token_position': args.class_token_position,
        'dataset': args.dataset,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    metadata_path = osp.join(epoch_dir, 'metadata.pth')
    torch.save(metadata, metadata_path)
    
    print(f"Saved checkpoint for epoch {epoch} to {epoch_dir}")


def train_prompts(prompt_learner, unet, vae, scheduler, train_loader, args):
    """Train the learnable prompts using CoOp approach with differentiable diffusion loss"""
    optimizer = optim.SGD(prompt_learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    
    # Create save directory with saving enabled
    save_dir = None
    if hasattr(args, 'save_folder') and args.save_folder:
        save_dir = osp.join(DATASET_ROOT, 'prompts' , args.save_folder)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Will save checkpoints to: {save_dir}")
    
    print("Training learnable prompts...")
    for epoch in range(args.max_epoch):
        prompt_learner.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            print(images.shape, labels.shape)
            
            # Encode images to latents
            with torch.no_grad():
                if args.dtype == 'float16':
                    images = images.half()
                x0 = vae.encode(images).latent_dist.mean
                x0 *= 0.18215
            
            # Get current prompt embeddings (this is differentiable)
            text_embeddings = prompt_learner()
            
            # Compute classification loss for the batch
            batch_loss = 0
            batch_correct = 0
            
            for i, (latent, label) in enumerate(zip(x0, labels)):
                # Use differentiable evaluation
                class_probs, class_errors = eval_prob_adaptive_differentiable(
                    unet, latent.unsqueeze(0), text_embeddings, scheduler, args, 
                    args.img_size // 8
                )
                
                # Cross-entropy loss using the class probabilities
                target = torch.tensor([label], device=device)
                loss = F.cross_entropy(class_probs.unsqueeze(0), target)
                batch_loss += loss
                
                # For accuracy calculation
                pred_idx = torch.argmax(class_probs).item()
                if pred_idx == label.item():
                    batch_correct += 1
            
            batch_loss = batch_loss / len(images)
            batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            correct += batch_correct
            total += len(images)
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.6f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        scheduler_lr.step()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, "
              f"Acc = {100*correct/total:.2f}%")
        
        # Save checkpoint after each epoch
        if save_dir is not None:
            save_epoch_checkpoints(prompt_learner, epoch + 1, save_dir, args)


def main():
    parser = argparse.ArgumentParser()

    # Original dataset args
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # Original run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # CoOp training args
    parser.add_argument('--n_ctx', type=int, default=16, help='Number of context tokens')
    parser.add_argument('--ctx_init', type=str, default='a blurry photo of a', help='Initial context')
    parser.add_argument('--class_token_position', type=str, default='end', 
                        choices=['end', 'middle', 'front'], help='Position of class token')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=50, help='Maximum training epochs')
    
    # Folder saving argument
    parser.add_argument('--save_folder', type=str, default='learned_prompts1', 
                        help='Folder name in DATASET_ROOT to save epoch checkpoints')

    args = parser.parse_args()

    # Set up dataset and transforms
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    
    # Load datasets
    train_dataset = get_target_dataset(args.dataset, train=True, transform=transform)
    test_dataset = get_target_dataset(args.dataset, train=False, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=4)
    
    # Load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in unet.parameters():
        param.requires_grad = False
    
    class_names = train_dataset.classes 

    # Initialize prompt learner (now requires text_encoder for proper initialization)
    prompt_learner = PromptLearner(class_names, tokenizer, text_encoder, args.n_ctx, 
                                 args.ctx_init, args.class_token_position).to(device)
    
    train_prompts(prompt_learner, unet, vae, scheduler, train_loader, args)


if __name__ == '__main__':
    main()