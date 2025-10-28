import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageSegmentation
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 43

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(42)

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


class MultiObjectDataset:
    """Custom dataset for multi-object images with position-based subfolders."""
    
    def __init__(self, root_dir, num_objects, positions, transform=None):
        """
        Args:
            root_dir: Base directory (e.g., '/path/to/comco')
            num_objects: Number of objects (e.g., 2, 3, 4, 5)
            positions: List of position folders to use (e.g., ['left', 'right'])
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir) / str(num_objects)
        self.transform = transform
        self.image_data = []  # Will store (path, objects, position, biggest_idx, filename)
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory {self.root_dir} does not exist")
        
        # Process each position subfolder
        for position in positions:
            position_dir = self.root_dir / position
            print(f"Loading images from {position_dir}...")
            if not position_dir.exists():
                print(f"Warning: {position_dir} does not exist, skipping")
                continue
            
            image_paths = sorted(list(position_dir.glob("*.png")) + list(position_dir.glob("*.jpg")))
            
            for img_path in image_paths:
                filename = img_path.stem
                # Parse filename: folder2_keyboard_backpack_motor_airplane_3.png
                # Format: folder{num}_obj1_obj2_..._objN_{biggest_idx}
                parts = filename.split('_')
                
                # Last part is the biggest object index (0-based)
                try:
                    biggest_idx = int(parts[-1])-1
                except ValueError:
                    print(f"Warning: Could not parse biggest object index from {filename}, skipping")
                    continue
                
                # Objects are all parts between folder prefix and biggest_idx
                # Skip first part (folderX) and last part (biggest_idx)
                objects = parts[1:-1]
                
                if len(objects) != num_objects:
                    print(f"Warning: {filename} has {len(objects)} objects but expected {num_objects}, skipping")
                    continue
                
                if biggest_idx >= len(objects):
                    print(f"Warning: {filename} has biggest_idx={biggest_idx} but only {len(objects)} objects, skipping")
                    continue
                
                self.image_data.append({
                    'path': img_path,
                    'objects': objects,
                    'position': position,
                    'biggest_idx': biggest_idx,
                    'filename': filename
                })
        
        if len(self.image_data) == 0:
            raise ValueError(f"No images found in {self.root_dir}/{{{','.join(positions)}}}")
        
        # Extract all unique objects from all filenames
        all_objects_set = set()
        for data in self.image_data:
            all_objects_set.update(data['objects'])
        
        self.all_objects = sorted(list(all_objects_set))
        
        print(f"Found {len(self.image_data)} images total:")
        for position in positions:
            count = sum(1 for d in self.image_data if d['position'] == position)
            print(f"  - {position}: {count}")
        print(f"Found {len(self.all_objects)} unique objects: {self.all_objects}")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        image = Image.open(data['path'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, data['objects'], data['position'], data['biggest_idx'], data['filename']


def create_prompts_for_scenario(dataset, scenario_mode, num_objects):
    """
    Create positive and negative prompts for each image based on number of objects.
    
    For 2 objects:
        Scenario 1: Pos: biggest first; Neg: wrong first
        Scenario 2: Pos: biggest last; Neg: biggest first, wrong last
    
    For 3 objects:
        Scenario 1: Pos: biggest first; Neg: wrong first, other two unchanged
        Scenario 2: Pos: biggest last; Neg: biggest first, wrong at position 3
    
    For 4 objects:
        Scenario 1: Pos: biggest first; Neg: wrong first, other three unchanged
        Scenario 2: Pos: biggest last; Neg: biggest first, wrong at position 4
    
    For 5 objects:
        Scenario 1: Pos: biggest first; Neg: wrong first, other four unchanged
        Scenario 2: Pos: biggest last; Neg: biggest first, wrong at position 5
    """
    prompt_data = []
    all_objects = dataset.all_objects
    
    for idx in range(len(dataset)):
        _, objects, position, biggest_idx, filename = dataset[idx]
        
        biggest_object = objects[biggest_idx]
        other_objects = [obj for i, obj in enumerate(objects) if i != biggest_idx]
        
        # Select a wrong object that is NOT in the current image
        available_wrong_objects = [obj for obj in all_objects if obj not in objects]
        
        if not available_wrong_objects:
            print(f"Warning: {filename} - no available wrong objects, skipping")
            continue
        
        wrong_object = random.choice(available_wrong_objects)
        
        # Create prompts based on number of objects
        if num_objects == 2:
            if scenario_mode == 1:
                # Positive: biggest first
                positive_prompt = f"a photo of a {biggest_object} and a {other_objects[0]}"
                # Negative: wrong object first
                negative_prompt = f"a photo of a {wrong_object} and a {other_objects[0]}"
                
            elif scenario_mode == 2:
                # Positive: biggest last
                positive_prompt = f"a photo of a {other_objects[0]} and a {biggest_object}"
                # Negative: biggest first, wrong last
                negative_prompt = f"a photo of a {biggest_object} and a {wrong_object}"
        
        elif num_objects == 3:
            if len(other_objects) < 2:
                print(f"Warning: {filename} needs at least 3 objects, skipping")
                continue
                
            if scenario_mode == 1:
                # Positive: biggest first
                positive_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, and a {other_objects[1]}"
                # Negative: wrong first, other two unchanged
                negative_prompt = f"a photo of a {wrong_object}, a {other_objects[0]}, and a {other_objects[1]}"
                
            elif scenario_mode == 2:
                # Positive: biggest last
                positive_prompt = f"a photo of a {other_objects[0]}, a {other_objects[1]}, and a {biggest_object}"
                # Negative: biggest first, wrong at position 3
                negative_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, and a {wrong_object}"
        
        elif num_objects == 4:
            if len(other_objects) < 3:
                print(f"Warning: {filename} needs at least 4 objects, skipping")
                continue
                
            if scenario_mode == 1:
                # Positive: biggest first
                positive_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, a {other_objects[1]}, and a {other_objects[2]}"
                # Negative: wrong first, other three unchanged
                negative_prompt = f"a photo of a {wrong_object}, a {other_objects[0]}, a {other_objects[1]}, and a {other_objects[2]}"
                
            elif scenario_mode == 2:
                # Positive: biggest last
                positive_prompt = f"a photo of a {other_objects[0]}, a {other_objects[1]}, a {other_objects[2]}, and a {biggest_object}"
                # Negative: biggest first, wrong at position 4
                negative_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, a {other_objects[1]}, and a {wrong_object}"
        
        elif num_objects == 5:
            if len(other_objects) < 4:
                print(f"Warning: {filename} needs at least 5 objects, skipping")
                continue
                
            if scenario_mode == 1:
                # Positive: biggest first
                positive_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, a {other_objects[1]}, a {other_objects[2]}, and a {other_objects[3]}"
                # Negative: wrong first, other four unchanged
                negative_prompt = f"a photo of a {wrong_object}, a {other_objects[0]}, a {other_objects[1]}, a {other_objects[2]}, and a {other_objects[3]}"
                
            elif scenario_mode == 2:
                # Positive: biggest last
                positive_prompt = f"a photo of a {other_objects[0]}, a {other_objects[1]}, a {other_objects[2]}, a {other_objects[3]}, and a {biggest_object}"
                # Negative: biggest first, wrong at last position
                negative_prompt = f"a photo of a {biggest_object}, a {other_objects[0]}, a {other_objects[1]}, a {other_objects[2]}, and a {wrong_object}"
        
        else:
            raise ValueError(f"Unsupported number of objects: {num_objects}")
        
        prompt_data.append({
            'filename': filename,
            'position': position,
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'objects': '_'.join(objects),
            'biggest_object': biggest_object,
            'biggest_idx': biggest_idx,
            'wrong_object': wrong_object
        })
    
    return pd.DataFrame(prompt_data)


class SaveAttnProcessor:
    """Attention processor that saves cross-attention maps."""
    def __init__(self, store):
        self.store = store
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store cross-attention
        if is_cross:
            self.store.append(attention_probs.detach())
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class MultiObjectDiffusionEvaluator:
    def __init__(self, args, scenario_mode, num_objects, positions):
        self.args = args
        self.device = device
        self.scenario_mode = scenario_mode
        self.num_objects = num_objects
        self.positions = positions
        
        # Initialize tracking variables for results
        self.all_predictions = []
        self.all_labels = []
        self.results_details = []
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        self._setup_run_folder()
        self._setup_prompts()
        self._setup_noise()

        
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
        self.target_dataset = MultiObjectDataset(
            self.args.dataset_path, 
            self.num_objects,
            self.positions,
            transform=transform
        )
        
    def _setup_prompts(self):
        # Create prompts based on scenario mode and number of objects
        self.prompts_df = create_prompts_for_scenario(
            self.target_dataset, 
            self.scenario_mode,
            self.num_objects
        )
        
        # Save prompts for reference
        if osp.exists(osp.join(self.run_folder, 'prompts.csv')):
            print("Warning: prompts.csv already exists and will be used")
        else:
            prompts_save_path = osp.join(self.run_folder, 'prompts.csv')
            self.prompts_df.to_csv(prompts_save_path, index=False)
            print(f"Prompts saved to {prompts_save_path}")
        
        # Encode prompts
        all_prompts = []
        for _, row in self.prompts_df.iterrows():
            all_prompts.append(row['positive_prompt'])
            all_prompts.append(row['negative_prompt'])
        
        self.all_prompt_texts = all_prompts
        
        text_input = self.tokenizer(all_prompts, padding="max_length",
                            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_input.input_ids), 100):
                text_embeddings = self.text_encoder(
                    text_input.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        self.text_embeddings = torch.cat(embeddings, dim=0)
        assert len(self.text_embeddings) == len(all_prompts)
        
    def _setup_noise(self):
        if self.args.noise_path is not None:
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        name = f"numobj{self.num_objects}_pos{'_'.join(self.positions)}_"
        name += f"scenario{self.scenario_mode}_v{self.args.version}_{self.args.n_trials}trials_"
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
        if self.args.visualize:
            name += '_viz'
        if self.args.extra is not None:
            name += f'_{self.args.extra}'
        
        self.run_folder = osp.join(LOG_DIR, 'multi_object', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')
    
    
    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, latent_size=64, 
                          all_noise=None, visualize=False, original_image=None, 
                          prompt_texts=None, objects_str=None):
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
            
            if visualize and original_image is not None:
                pred_errors = self.eval_error_visualize(
                    unet, scheduler, self.vae, self.tokenizer, original_image, latent, 
                    all_noise, ts, noise_idxs, text_embeds, text_embed_idxs, 
                    objects_str, args.batch_size, args.dtype, args.loss, T, 
                    visualize=True, prompts=prompt_texts
                )
            else:
                pred_errors = self.eval_error(
                    unet, scheduler, latent, all_noise, ts, noise_idxs,
                    text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss
                )
            
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
            assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
            pred_errors = torch.zeros(len(ts), device='cpu')
            idx = 0
            
            if visualize:
                viz_data = defaultdict(lambda: {
                    'images': [], 
                    'errors': [], 
                    'timesteps': [],
                    'attention_maps': []
                })
            
            with torch.inference_mode():
                for batch_idx in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                    batch_ts = torch.tensor(ts[idx: idx + batch_size])
                    batch_text_idxs = text_embed_idxs[idx: idx + batch_size]
                    noise = all_noise[noise_idxs[idx: idx + batch_size]]
                    
                    alpha_prod_t = scheduler.alphas_cumprod[batch_ts]
                    sqrt_alpha_prod = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1).to(self.device)
                    sqrt_one_minus_alpha_prod = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1).to(self.device)
                    
                    noised_latent = latent * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
                    
                    t_input = batch_ts.to(self.device).half() if dtype == 'float16' else batch_ts.to(self.device)
                    text_input = text_embeds[batch_text_idxs]
                    if dtype == 'float16':
                        text_input = text_input.half()
                        noised_latent = noised_latent.half()

                    attention_store = []
                    unet.set_attn_processor(SaveAttnProcessor(attention_store))

                    noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                    
                    denoised_latent = (noised_latent - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                    reconstructed_image = vae.decode(denoised_latent / vae.config.scaling_factor).sample
                
                    if dtype == 'float16':
                        reconstructed_image = reconstructed_image.float()
                    
                    if loss == 'l2':
                        error = F.mse_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                    elif loss == 'l1':
                        error = F.l1_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                    elif loss == 'huber':
                        error = F.huber_loss(original_image, reconstructed_image, reduction='none').mean(dim=1)
                    else:
                        raise NotImplementedError
                    
                    pred_errors[idx: idx + len(batch_ts)] = error.mean(dim=(1, 2)).detach().cpu()                        
                    
                    if visualize:
                        for i in range(len(batch_ts)):
                            prompt_idx = batch_text_idxs[i]
                            viz_data[prompt_idx]['images'].append(reconstructed_image[i].detach().cpu())
                            viz_data[prompt_idx]['errors'].append(error[i].detach().cpu())
                            viz_data[prompt_idx]['timesteps'].append(batch_ts[i].item())
                            viz_data[prompt_idx]['attention_maps'].append([attn[i:i+1].detach().cpu() for attn in attention_store])
                    
                    idx += len(batch_ts)
            
            if visualize and len(viz_data) > 0:
                timestamp = datetime.now().strftime("%H%M%S")
                
                for prompt_idx, data in viz_data.items():
                    prompt_text = prompts[prompt_idx] if prompts is not None else f"Prompt {prompt_idx}"
                    
                    recon_images = torch.stack(data['images'])
                    error_maps = torch.stack(data['errors'])
                    timesteps = torch.tensor(data['timesteps'])
                    
                    save_path = osp.join(self.run_folder, f'error_heatmap_prompt{prompt_idx}_{timestamp}.png')
                    self.visualize_error_heatmap(
                        original_image, target_fruit, recon_images, error_maps, 
                        timesteps, prompt_text, save_path=save_path
                    )
                    
                    p = prompt_text[:-1].lower() if prompt_text.endswith('.') else prompt_text.lower()
                    tokens_to_vis = p.split()[4], p.split()[7]
                    
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
        """Visualize attention maps across all timesteps in a grid layout."""
        tokens = tokenizer.tokenize(prompt)
        tokens = [t.replace("</w>", "") for t in tokens]
        
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
        
        sort_idx = torch.argsort(timesteps)
        timesteps = timesteps[sort_idx]
        attention_maps = [attention_maps[i] for i in sort_idx]
        
        image_to_plot = image[0] if image.ndim == 4 else image
        image_np = image_to_plot.permute(1, 2, 0).cpu().numpy()
        
        if image_np.dtype not in ['float32', 'float64']:
            image_np = image_np.astype('float32')
        if image_np.min() < 0:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        num_timesteps = len(attention_maps)
        num_tokens = len(token_indices)
        max_cols = 6
        num_rows = (num_timesteps + max_cols - 1) // max_cols
        num_cols = min(num_timesteps, max_cols)
        
        for token_word, token_word_ids in token_indices:
            fig = plt.figure(figsize=(5 * num_cols, 10 * num_rows))
            gs = fig.add_gridspec(num_rows + 1, num_cols, hspace=0.3, wspace=0.3)
            
            ax_orig = fig.add_subplot(gs[0, :])
            ax_orig.imshow(image_np)
            ax_orig.set_title(f'Original Image\nPrompt: "{prompt}"\nToken: "{token_word}"', 
                            fontsize=12, fontweight='bold')
            ax_orig.axis('off')
            
            for idx, (attn_store, t) in enumerate(zip(attention_maps, timesteps)):
                row = (idx // num_cols) + 1
                col = idx % num_cols
                
                resized = []
                for attn in attn_store:
                    if attn.dim() == 4:
                        attn = attn.squeeze(0)
                    
                    heads, HW, tokens = attn.shape
                    h = w = int(HW ** 0.5)
                    attn_map = attn.view(heads, h, w, tokens).permute(0, 3, 1, 2)
                    attn_map = F.interpolate(attn_map, size=(64, 64), mode="bilinear", align_corners=False)
                    resized.append(attn_map)
                
                attn_all = torch.cat(resized, dim=0)
                attn_mean = attn_all.mean(0)
                
                heatmap = attn_mean[token_word_ids].mean(0)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                heatmap_up = heatmap.unsqueeze(0).unsqueeze(0)
                heatmap_up = F.interpolate(heatmap_up, size=upsample_size, mode="bilinear", align_corners=False)
                heatmap_up = heatmap_up.squeeze().cpu().numpy()
                
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(image_np)
                im = ax.imshow(heatmap_up, cmap='jet', alpha=0.6, interpolation='nearest')
                ax.axis('off')
                ax.set_title(f't={t.item()}\nAttn: {heatmap.mean():.4f}', fontsize=10)
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            
            plt.suptitle(f'Cross-Attention Maps for "{token_word}" Across All Timesteps', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            if save_path:
                base_path = save_path.replace('.png', '')
                token_save_path = f"{base_path}_{token_word}.png"
                plt.savefig(token_save_path, dpi=150, bbox_inches='tight')
                print(f"Attention visualization for '{token_word}' saved to {token_save_path}")
                plt.close()
        return fig
    
    def visualize_error_heatmap(self, original_image, target_objects, reconstructed_images, 
                            error_maps, timesteps, prompt_text, save_path=None):
        """Visualize original image, reconstructed images, and error heatmaps for all timesteps."""
        num_samples = len(reconstructed_images)
        
        sort_idx = torch.argsort(timesteps)
        timesteps = timesteps[sort_idx]
        reconstructed_images = reconstructed_images[sort_idx]
        error_maps = error_maps[sort_idx]
        
        max_cols = 6
        num_rows = (num_samples + max_cols - 1) // max_cols
        num_cols = min(num_samples, max_cols)
        
        fig = plt.figure(figsize=(5 * num_cols, 10 * num_rows))
        gs = fig.add_gridspec(num_rows + 1, num_cols, hspace=0.3, wspace=0.3)
        
        orig_img = ((original_image[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1).astype(np.float32)
        
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f'Original Image: {target_objects}\nPrompt: "{prompt_text}"', 
                        fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        for i in range(num_samples):
            row = (i // num_cols) + 1
            col = i % num_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            recon_img = ((reconstructed_images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1).astype(np.float32)
            error_map = error_maps[i].cpu().numpy().astype(np.float32)
            
            ax.imshow(recon_img, extent=[0, 1, 0.5, 1])
            im = ax.imshow(error_map, cmap='hot', interpolation='nearest', 
                        extent=[0, 1, 0, 0.5], aspect='auto')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f't={timesteps[i].item()}\nAvg Error: {error_map.mean():.4f}', 
                        fontsize=10)
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Reconstruction Error Analysis Across All Timesteps\nTarget: {target_objects}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            plt.close()
        
        return fig

    def save_results_summary(self):
        """Save comprehensive summary of results."""
        if len(self.all_predictions) == 0:
            print("No results to summarize")
            return
            
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Multi-Object Diffusion Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            correct = sum(1 for p in self.all_predictions if p == 0)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Number of objects: {self.num_objects}\n")
            f.write(f"Positions: {', '.join(self.positions)}\n")
            f.write(f"Scenario Mode: {self.scenario_mode}\n")
            if self.scenario_mode == 1:
                f.write("  - Positive: biggest object first\n")
                f.write("  - Negative: wrong object first, others unchanged\n")
            else:
                f.write("  - Positive: biggest object last\n")
                f.write("  - Negative: biggest object first, wrong at last position\n")
            
            f.write(f"\nOverall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct (chose positive prompt): {correct}\n")
            f.write(f"Incorrect (chose negative prompt): {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            f.write("Results by Position:\n")
            for position in self.positions:
                position_results = [r for r in self.results_details if r.get('position') == position]
                if position_results:
                    pos_correct = sum(1 for r in position_results if r['chose_positive'])
                    pos_total = len(position_results)
                    pos_acc = (pos_correct / pos_total * 100) if pos_total > 0 else 0
                    f.write(f"  {position}: {pos_correct}/{pos_total} ({pos_acc:.2f}%)\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Visualization: {self.args.visualize}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Loss function: {self.args.loss}\n")
        
        print(f"Results summary saved to {summary_path}")
        
        results_df = pd.DataFrame(self.results_details)
        results_csv_path = osp.join(self.run_folder, 'detailed_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to {results_csv_path}")

    def run_evaluation(self):
        formatstr = get_formatstr(len(self.target_dataset) - 1)
        correct = 0
        total = 0
        pbar = tqdm.tqdm(range(len(self.target_dataset)))
        
        for i in pbar:
            if total > 0:
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print(f'Skipping {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['chose_positive'])
                    total += 1
                    self.all_predictions.append(0 if data['chose_positive'] else 1)
                    self.all_labels.append(0)
                continue
            
            image, objects, position, biggest_idx, filename = self.target_dataset[i]
            
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215

            text_embeddings_pair = self.text_embeddings[i*2:(i+1)*2]
            prompt_texts_pair = [self.all_prompt_texts[i*2], self.all_prompt_texts[i*2+1]]
            objects_str = '_'.join(objects)
                
            _, pred_idx, pred_errors = self.eval_prob_adaptive(
                self.unet, x0, text_embeddings_pair, self.scheduler, 
                self.args, self.latent_size, self.all_noise,
                visualize=self.args.visualize,
                original_image=img_input,
                prompt_texts=prompt_texts_pair,
                objects_str=objects_str
            )
            
            chose_positive = (pred_idx == 0)
            
            result_detail = {
                'filename': filename,
                'position': position,
                'objects': '_'.join(objects),
                'biggest_idx': biggest_idx,
                'chose_positive': chose_positive,
                'positive_loss': pred_errors[pred_idx]['pred_errors'].mean().item(),
                'negative_loss': pred_errors[1-pred_idx]['pred_errors'].mean().item(),
                'positive_prompt': self.prompts_df.iloc[i]['positive_prompt'],
                'negative_prompt': self.prompts_df.iloc[i]['negative_prompt']
            }
            self.results_details.append(result_detail)
            
            torch.save({
                'errors': pred_errors,
                'chose_positive': chose_positive,
                'filename': filename,
                'position': position,
                'objects': objects,
                'biggest_idx': biggest_idx
            }, fname)
            
            if chose_positive:
                correct += 1
            total += 1
            
            self.all_predictions.append(0 if chose_positive else 1)
            self.all_labels.append(0)
        
        self.save_results_summary()


class MultiConfigRunner:
    """Runner for multiple test configurations."""
    
    def __init__(self, args):
        self.args = args
        self.all_results = []
        
    def run_all_configurations(self):
        """Run evaluation for all combinations of scenario_modes, num_objects, and positions."""
        # Generate all combinations
        configs = list(itertools.product(
            self.args.scenario_modes,
            self.args.num_objects_list,
            [tuple(self.args.positions)]
        ))
        
        print(f"\n{'='*70}")
        print(f"Running {len(configs)} total configurations")
        print(f"Scenario modes: {self.args.scenario_modes}")
        print(f"Number of objects: {self.args.num_objects_list}")
        print(f"Positions: {self.args.positions}")
        print(f"{'='*70}\n")
        
        for config_idx, (scenario_mode, num_objects, positions) in enumerate(configs, 1):
            print(f"\n{'='*70}")
            print(f"Configuration {config_idx}/{len(configs)}")
            print(f"Scenario Mode: {scenario_mode}, Objects: {num_objects}, Positions: {positions}")
            print(f"{'='*70}\n")
            
            try:
                evaluator = MultiObjectDiffusionEvaluator(
                    self.args, 
                    scenario_mode, 
                    num_objects, 
                    list(positions)
                )
                evaluator.run_evaluation()
                
                # Collect results
                correct = sum(1 for p in evaluator.all_predictions if p == 0)
                total = len(evaluator.all_predictions)
                accuracy = (correct / total * 100) if total > 0 else 0
                
                config_result = {
                    'scenario_mode': scenario_mode,
                    'num_objects': num_objects,
                    'positions': '_'.join(positions),
                    'total_samples': total,
                    'correct': correct,
                    'accuracy': accuracy,
                    'run_folder': evaluator.run_folder
                }
                self.all_results.append(config_result)
                
                print(f"\nConfiguration {config_idx} completed: {accuracy:.2f}% accuracy")
                
            except Exception as e:
                print(f"Error in configuration {config_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.save_aggregate_summary()
    
    def save_aggregate_summary(self):
        """Save aggregate summary across all configurations."""
        if not self.all_results:
            print("No results to aggregate")
            return
        
        # Create main results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aggregate_folder = osp.join(LOG_DIR, 'multi_object', f'aggregate_results_{timestamp}')
        os.makedirs(aggregate_folder, exist_ok=True)
        
        # Save detailed CSV
        results_df = pd.DataFrame(self.all_results)
        csv_path = osp.join(aggregate_folder, 'aggregate_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nAggregate results saved to {csv_path}")
        
        # Save text summary
        summary_path = osp.join(aggregate_folder, 'aggregate_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("AGGREGATE RESULTS ACROSS ALL CONFIGURATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            total_samples = sum(r['total_samples'] for r in self.all_results)
            total_correct = sum(r['correct'] for r in self.all_results)
            overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total configurations tested: {len(self.all_results)}\n")
            f.write(f"  Total samples evaluated: {total_samples}\n")
            f.write(f"  Total correct predictions: {total_correct}\n")
            f.write(f"  Overall accuracy: {overall_accuracy:.2f}%\n\n")
            
            # Results by scenario mode
            f.write("Results by Scenario Mode:\n")
            for scenario in sorted(set(r['scenario_mode'] for r in self.all_results)):
                scenario_results = [r for r in self.all_results if r['scenario_mode'] == scenario]
                scenario_correct = sum(r['correct'] for r in scenario_results)
                scenario_total = sum(r['total_samples'] for r in scenario_results)
                scenario_acc = (scenario_correct / scenario_total * 100) if scenario_total > 0 else 0
                f.write(f"  Scenario {scenario}: {scenario_correct}/{scenario_total} ({scenario_acc:.2f}%)\n")
            f.write("\n")
            
            # Results by number of objects
            f.write("Results by Number of Objects:\n")
            for num_obj in sorted(set(r['num_objects'] for r in self.all_results)):
                obj_results = [r for r in self.all_results if r['num_objects'] == num_obj]
                obj_correct = sum(r['correct'] for r in obj_results)
                obj_total = sum(r['total_samples'] for r in obj_results)
                obj_acc = (obj_correct / obj_total * 100) if obj_total > 0 else 0
                f.write(f"  {num_obj} objects: {obj_correct}/{obj_total} ({obj_acc:.2f}%)\n")
            f.write("\n")
            
            # Results by position
            f.write("Results by Position Configuration:\n")
            for pos_config in sorted(set(r['positions'] for r in self.all_results)):
                pos_results = [r for r in self.all_results if r['positions'] == pos_config]
                pos_correct = sum(r['correct'] for r in pos_results)
                pos_total = sum(r['total_samples'] for r in pos_results)
                pos_acc = (pos_correct / pos_total * 100) if pos_total > 0 else 0
                f.write(f"  {pos_config}: {pos_correct}/{pos_total} ({pos_acc:.2f}%)\n")
            f.write("\n")
            
            # Detailed results table
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS BY CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Scenario':<10} {'Objects':<10} {'Positions':<15} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in self.all_results:
                f.write(f"{result['scenario_mode']:<10} "
                       f"{result['num_objects']:<10} "
                       f"{result['positions']:<15} "
                       f"{result['total_samples']:<10} "
                       f"{result['correct']:<10} "
                       f"{result['accuracy']:<10.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Results saved at: {timestamp}\n")
            f.write(f"Individual run folders:\n")
            for result in self.all_results:
                f.write(f"  - {result['run_folder']}\n")
        
        print(f"Aggregate summary saved to {summary_path}")
        print(f"\n{'='*80}")
        print(f"ALL CONFIGURATIONS COMPLETED")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"Results folder: {aggregate_folder}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()

    # Dataset args - now accept multiple values
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/comco',
                        help='Path to dataset base directory (containing 2,3,4,5 folders)')
    parser.add_argument('--num_objects_list', nargs='+', type=int, default=[2], 
                        choices=[2, 3, 4, 5],
                        help='List of object numbers to test (e.g., --num_objects_list 2 3 4)')
    parser.add_argument('--positions', nargs='+', type=str, default=['left'],
                        choices=['left', 'right', 'middle', 'up', 'down'],
                        help='Position subfolders to use (e.g., --positions left right)')
    parser.add_argument('--scenario_modes', nargs='+', type=int, default=[2], 
                        choices=[1, 2],
                        help='Scenario modes to test (e.g., --scenario_modes 1 2)')

    # Model args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'))
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='Extra string for run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'))
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization of attention maps and error heatmaps')
    parser.add_argument('--run_folder', type=str, default=None, help='If set, use this folder to save results')

    # Adaptive sampling args
    parser.add_argument('--to_keep', nargs='+', type=int, default=[1])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[50])

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # Run all configurations
    runner = MultiConfigRunner(args)
    runner.run_all_configurations()


if __name__ == '__main__':
    main()