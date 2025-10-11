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

# All possible fruits for classification
ALL_FRUITS = ['cherry', 'pomegranate', 'strawberry', 'tomato', 'banana', 
              'lemon', 'corn', 'broccoli', 'cucumber', 'brinjal', 'plum', 
              'orange', 'carrot']


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


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def parse_compound_filename(filename):
    """
    Parse compound fruit filename.
    Expected format: fruit1_fruit2_number.jpg
    Example: banana_brinjal_220.jpg
    
    Returns: (fruit1, fruit2) or None if parsing fails
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    if len(parts) >= 3:
        fruit1 = parts[0]
        fruit2 = parts[1]
        if fruit1 in ALL_FRUITS and fruit2 in ALL_FRUITS:
            return fruit1, fruit2
    
    return None


def parse_single_filename(filename):
    """
    Parse single fruit filename.
    Expected format: fruit_number.jpg
    Example: banana_006.jpg
    Returns: fruit or None if parsing fails
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    if len(parts) >= 1:
        fruit = parts[0]
        if fruit in ALL_FRUITS:
            return fruit
    
    return None


def parse_compound_unnatural_filename(filename):
    """
    Parse unnatural fruit combination filename.
    Expected format: compound_fruit1_color1_fruit2_color2_number.png
    Example: compound_banana_green_brinjal_red_220.png
    
    Returns: (fruit1, fruit2) or None if parsing fails
    Note: We only extract fruits, not colors, since we're classifying fruits
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    # Expected: ['compound', fruit1, color1, fruit2, color2, number]
    if len(parts) >= 6 and parts[0] == 'compound':
        fruit1 = parts[1]
        fruit2 = parts[3]  # Skip color at parts[2]
        if fruit1 in ALL_FRUITS and fruit2 in ALL_FRUITS:
            return fruit1, fruit2
    
    return None


def parse_single_unnatural_filename(filename):
    """
    Parse single unnatural fruit filename.
    Expected format: fruit_color_number1_number2.png
    Example: banana_green_006_202011.png
    Returns: fruit or None if parsing fails
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    # Expected: [fruit, color, number1, number2]
    if len(parts) >= 2:
        fruit = parts[0]
        if fruit in ALL_FRUITS:
            return fruit
    
    return None

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
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
        attn_probs  = torch.nn.functional.softmax(attn_scores, dim=-1)

        if encoder_hidden_states is not None:
            self.store.append(attn_probs.detach().cpu())

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class FruitDataset:
    """Custom dataset for fruit images"""
    def __init__(self, root_dir, mode='compound', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'compound':
            self.image_paths = glob.glob(osp.join(root_dir, "compound", "*.jpg"))
        elif mode == 'single':
            self.image_paths = glob.glob(osp.join(root_dir, "single", "*.jpg"))
        elif mode == 'compound_unnatural':
            self.image_paths = glob.glob(osp.join(root_dir, "compound_unnatural", "*.jpg"))
            self.image_paths += glob.glob(osp.join(root_dir, "compound_unnatural", "*.png"))
        elif mode == 'single_unnatural':
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
        
        if self.mode == 'compound':
            parsed = parse_compound_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruit1, fruit2 = parsed
                fruits = [fruit1, fruit2]
        elif self.mode == 'compound_unnatural':
            parsed = parse_compound_unnatural_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruit1, fruit2 = parsed
                fruits = [fruit1, fruit2]
        elif self.mode == 'single_unnatural':
            fruit = parse_single_unnatural_filename(filename)
            if fruit is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruits = [fruit]
        else:  # single mode
            fruit = parse_single_filename(filename)
            if fruit is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruits = [fruit]
                
        return image, fruits, image_path


class DiffusionFruitClassifier:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0
        self.other_fruit_predictions = 0
        self.other_mistakes = 0
        self.total_classifications = 0
        
        # Confusion matrix
        self.confusion_matrix = np.zeros((len(ALL_FRUITS), len(ALL_FRUITS)), dtype=int)
        self.fruit_to_idx = {fruit: idx for idx, fruit in enumerate(ALL_FRUITS)}
        
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
        self.target_dataset = FruitDataset(self.args.data_folder, mode=self.args.mode, transform=transform)
        
    def _setup_noise(self):
        if self.args.noise_path is not None:
            assert not self.args.zero_noise
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        name = f"Fruit_classification_{self.args.mode}_v{self.args.version}_{self.args.n_trials}trials_"
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
            self.run_folder = osp.join(LOG_DIR, 'Fruit_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'Fruit', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def create_fruit_prompts(self):
        """Create prompts for all fruits"""
        prompts = []
        for fruit in ALL_FRUITS:
            prompt = f"A {fruit}."
            prompts.append(prompt)
        return prompts

    def classify_prediction(self, predicted_fruit, correct_fruit, other_fruit=None):
        """Classify the type of prediction made"""
        if predicted_fruit == correct_fruit:
            return 'correct'
        elif other_fruit is not None and predicted_fruit == other_fruit:
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

    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, image, target_info, 
                        latent_size=64, all_noise=None, prompts=None):
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

    def perform_fruit_classification(self, image, correct_fruit, other_fruit=None):
        """Perform fruit classification"""
        prompts = self.create_fruit_prompts()
        
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
            self.args, img_input, correct_fruit, self.latent_size, self.all_noise,
            prompts=prompts
        )
        
        predicted_fruit = ALL_FRUITS[pred_idx]
        prediction_type = self.classify_prediction(predicted_fruit, correct_fruit, other_fruit)
        
        return predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx

    def save_confusion_matrix(self):
        """Save confusion matrix visualization and CSV"""
        # Save as CSV
        df = pd.DataFrame(self.confusion_matrix, index=ALL_FRUITS, columns=ALL_FRUITS)
        csv_path = osp.join(self.run_folder, 'confusion_matrix.csv')
        df.to_csv(csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        # Create visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ALL_FRUITS, yticklabels=ALL_FRUITS,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Fruit')
        plt.ylabel('True Fruit')
        plt.title('Fruit Classification Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        fig_path = osp.join(self.run_folder, 'confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix visualization saved to {fig_path}")
        
        # Calculate and save per-class metrics
        metrics_path = osp.join(self.run_folder, 'per_fruit_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Per-Fruit Metrics\n")
            f.write("=" * 50 + "\n\n")
            
            for i, fruit in enumerate(ALL_FRUITS):
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
                
                f.write(f"{fruit.upper()}:\n")
                f.write(f"  Total samples: {total_true}\n")
                f.write(f"  Correct predictions: {true_positives}\n")
                f.write(f"  Precision: {precision:.2f}%\n")
                f.write(f"  Recall: {recall:.2f}%\n")
                f.write(f"  F1-Score: {f1:.2f}\n\n")
        
        print(f"Per-fruit metrics saved to {metrics_path}")

    def save_results_summary(self):
        """Save a summary of fruit classification results."""
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Fruit Classification Results Summary ({self.args.mode.title()} Mode)\n")
            f.write("==========================================================\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                other_fruit_acc = (self.other_fruit_predictions / self.total_classifications * 100)
                other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                
                f.write(f"Fruit Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct fruit predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.args.mode == 'compound':
                    f.write(f"Other fruit predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
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
            f.write(f"Adaptive sampling - n_samples: {self.args.n_samples}\n")
            f.write(f"Adaptive sampling - to_keep: {self.args.to_keep}\n\n")
            
            f.write(f"All possible fruits evaluated: {', '.join(ALL_FRUITS)}\n")
        
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
            
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {classification_idx}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        predicted_fruit = data['predicted_fruit']
                        correct_fruit = data['correct_fruit']
                        
                        # Update confusion matrix
                        true_idx = self.fruit_to_idx[correct_fruit]
                        pred_idx = self.fruit_to_idx[predicted_fruit]
                        self.confusion_matrix[true_idx, pred_idx] += 1
                        
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                other_fruit = fruit2 if target_fruit == fruit1 else fruit1
                predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx = self.perform_fruit_classification(
                    image, target_fruit, other_fruit
                )
                
                # Update confusion matrix
                true_idx = self.fruit_to_idx[correct_fruit]
                pred_idx_cm = self.fruit_to_idx[predicted_fruit]
                self.confusion_matrix[true_idx, pred_idx_cm] += 1
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_fruit=predicted_fruit,
                    correct_fruit=correct_fruit,
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
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                print(f'Skipping classification {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_fruit = data['predicted_fruit']
                    correct_fruit = data['correct_fruit']
                    
                    # Update confusion matrix
                    true_idx = self.fruit_to_idx[correct_fruit]
                    pred_idx = self.fruit_to_idx[predicted_fruit]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.other_mistakes += 1
                    self.total_classifications += 1
                continue
            
            predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx = self.perform_fruit_classification(
                image, fruit, None
            )
            
            # Update confusion matrix
            true_idx = self.fruit_to_idx[correct_fruit]
            pred_idx_cm = self.fruit_to_idx[predicted_fruit]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_fruit=predicted_fruit,
                correct_fruit=correct_fruit,
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
        elif self.args.mode == 'compound_unnatural':
            self.run_evaluation_compound_unnatural()
        elif self.args.mode == 'single_unnatural':
            self.run_evaluation_single_unnatural()
        else:
            self.run_evaluation_single()
        
        # Generate summary
        self.save_results_summary()
        
        # Save confusion matrix
        self.save_confusion_matrix()
        
        # Print final results
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Fruit Classification Results ({self.args.mode.title()} Mode):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct fruit predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.args.mode in ['compound', 'compound_unnatural']:
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                other_mistakes_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Other fruit predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)")
                print(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)")
            else:
                incorrect_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Incorrect predictions: {self.other_mistakes} ({incorrect_acc:.2f}%)")

    def run_evaluation_compound_unnatural(self):
        """Run evaluation for unnatural compound fruit images"""
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
            
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {classification_idx}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        predicted_fruit = data['predicted_fruit']
                        correct_fruit = data['correct_fruit']
                        
                        # Update confusion matrix
                        true_idx = self.fruit_to_idx[correct_fruit]
                        pred_idx = self.fruit_to_idx[predicted_fruit]
                        self.confusion_matrix[true_idx, pred_idx] += 1
                        
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                other_fruit = fruit2 if target_fruit == fruit1 else fruit1
                predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx = self.perform_fruit_classification(
                    image, target_fruit, other_fruit
                )
                
                # Update confusion matrix
                true_idx = self.fruit_to_idx[correct_fruit]
                pred_idx_cm = self.fruit_to_idx[predicted_fruit]
                self.confusion_matrix[true_idx, pred_idx_cm] += 1
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_fruit=predicted_fruit,
                    correct_fruit=correct_fruit,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=target_fruit,
                    fruits=[fruit1, fruit2],
                    prompts=prompts,
                    image_path=image_path,
                    classification_idx=classification_idx
                ), fname)

    def run_evaluation_single_unnatural(self):
        """Run evaluation for unnatural single fruit images"""
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
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                print(f'Skipping classification {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_fruit = data['predicted_fruit']
                    correct_fruit = data['correct_fruit']
                    
                    # Update confusion matrix
                    true_idx = self.fruit_to_idx[correct_fruit]
                    pred_idx = self.fruit_to_idx[predicted_fruit]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.other_mistakes += 1
                    self.total_classifications += 1
                continue
            
            predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx = self.perform_fruit_classification(
                image, fruit, None
            )
            
            # Update confusion matrix
            true_idx = self.fruit_to_idx[correct_fruit]
            pred_idx_cm = self.fruit_to_idx[predicted_fruit]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_fruit=predicted_fruit,
                correct_fruit=correct_fruit,
                prediction_type=prediction_type,
                pred_idx=pred_idx,
                target_fruit=fruit,
                fruits=[fruit],
                prompts=prompts,
                image_path=image_path,
                classification_idx=i
            ), fname)


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--data_folder', type=str,
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/color',
                        help='Path to data folder')
    parser.add_argument('--mode', type=str, default='single_unnatural', 
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

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # Create classifier and run
    classifier = DiffusionFruitClassifier(args)
    classifier.run_evaluation()


if __name__ == '__main__':
    main()