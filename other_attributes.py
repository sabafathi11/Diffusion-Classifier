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

# Define attributes for each object in each category
ATTRIBUTES = {
    "Part-Whole": {
        "Tree": "leafy",
        "Car": "wheeled",
        "Bird": "feathered", 
        "Fish": "finned", 
        "House": "windowed",
        "Airplane": "winged",
        "Flower": "petaled",
        "Book": "paged", 
        "Chair": "legged",
        "Cat": "tailed"
    },
    "Shape": {
        "Ball": "round",
        "Plate": "round",
        "Clock": "round",
        "Wheel": "round",
        "Coin": "round",
        "Box": "square",
        "Window": "square",
        "Book": "square",
        "Table": "square",
        "Dice": "square"
    },
    "Material & Texture": {
        "Table": "wooden",
        "Spoon": "silver",
        "Mug": "ceramic",
        "Blanket": "woolen",
        "Door": "wooden",
        "Shoe": "leather",
        "Bag": "fabric",
        "Candle": "wax",
        "Ring": "golden",
        "Statue": "marble"
    },
    "Size": {
        "Elephant": "big",
        "Whale": "big",
        "Truck": "big",
        "Building": "big",
        "Wind turbine": "big",
        "Ant": "small",
        "Mouse": "small",
        "Pebble": "small",
        "Key": "small",
        "Bird": "small"
    },
    "Temperature": {
        "Tea": "hot",
        "Coffee": "hot",
        "Soup": "hot",
        "Ice cube": "cold",
        "Water bottle": "cold",
        "Juice": "cold",
        "Fireplace": "hot",
        "Stove": "hot", 
        "Engine": "hot",
        "Ice cream": "cold"
    }
}

# Define similar attributes that shouldn't be paired
SIMILAR_ATTRIBUTES = {
    "Part-Whole": [
        {"leafy","petaled"},
        {"feathered", "finned"},
    ],
    "Shape": [
        {"circular"},
        {"rectangular"}
    ],
    "Size": [
        {"big"},
        {"small"}
    ],
    "Temperature": [
        {"hot"},
        {"cold"}
    ]
}

def normalize_name(name):
    """Normalize a name for comparison (lowercase, replace underscores with spaces)."""
    return name.lower().replace('_', ' ').strip()

def sanitize_filename(name):
    """Convert name to safe filename format (replace spaces with underscores)."""
    return name.replace(' ', '_')

def are_attributes_similar(attr1, attr2, category):
    """Check if two attributes are too similar to be paired."""
    if attr1 == attr2:
        return True
    
    if category in SIMILAR_ATTRIBUTES:
        for group in SIMILAR_ATTRIBUTES[category]:
            if attr1 in group and attr2 in group:
                return True
    
    return False

def get_dissimilar_attributes(target_attr, all_attributes, category):
    """Get only attributes that are dissimilar to the target attribute."""
    dissimilar = []
    for attr in all_attributes:
        if not are_attributes_similar(target_attr, attr, category):
            dissimilar.append(attr)
    return dissimilar

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

def create_lookup_tables(category):
    """Create bidirectional lookup tables for objects and attributes."""
    obj_lookup = {}
    attr_lookup = {}
    obj_to_attr = {}
    attr_to_objs = defaultdict(list)
    
    for obj_key, attr_val in ATTRIBUTES[category].items():
        obj_normalized = normalize_name(obj_key)
        attr_normalized = normalize_name(attr_val)
        
        obj_lookup[obj_normalized] = obj_key
        attr_lookup[attr_normalized] = attr_val
        obj_to_attr[obj_key] = attr_val
        attr_to_objs[attr_val].append(obj_key)
    
    return obj_lookup, attr_lookup, obj_to_attr, attr_to_objs


def parse_compound_filename(filename, category, obj_lookup, attr_lookup):
    """Parse compound filename to extract object names and attributes."""
    name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    parts = name.split('_')
    
    # Remove trailing number if present
    if parts[-1].isdigit():
        parts = parts[:-1]
    
    if len(parts) < 4:
        return None
    
    # Try to parse as: attr1_obj1_attr2_obj2
    # We need to find the split point between the two object-attribute pairs
    for split_point in range(2, len(parts) - 1):
        # First pair: parts[0:split_point]
        attr1_norm = normalize_name(parts[0])
        attr1 = attr_lookup.get(attr1_norm)
        
        if not attr1:
            continue
        
        # Try to match object1 from parts[1:split_point]
        obj1 = None
        for i in range(split_point, 1, -1):
            potential_obj1 = '_'.join(parts[1:i])
            obj1_norm = normalize_name(potential_obj1)
            obj1 = obj_lookup.get(obj1_norm)
            if obj1:
                attr2_start = i
                break
        
        if not obj1:
            continue
        
        # Second pair: parts[attr2_start:]
        attr2_norm = normalize_name(parts[attr2_start])
        attr2 = attr_lookup.get(attr2_norm)
        
        if not attr2:
            continue
        
        # Try to match object2 from remaining parts
        for i in range(len(parts), attr2_start + 1, -1):
            potential_obj2 = '_'.join(parts[attr2_start + 1:i])
            obj2_norm = normalize_name(potential_obj2)
            obj2 = obj_lookup.get(obj2_norm)
            
            if obj2:
                return obj1, attr1, obj2, attr2
    
    return None


def parse_single_filename(filename, category, obj_lookup, attr_lookup):
    """Parse single filename to extract object name and attribute."""
    name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    parts = name.split('_')
    
    # Remove trailing number if present
    if parts[-1].isdigit():
        parts = parts[:-1]
    
    if len(parts) < 2:
        return None
    
    # First part should be the attribute
    attr_norm = normalize_name(parts[0])
    attr = attr_lookup.get(attr_norm)
    
    if not attr:
        return None
    
    # Try to match the remaining parts as an object name
    # Start with all remaining parts, then try progressively fewer
    for i in range(len(parts), 1, -1):
        potential_obj = '_'.join(parts[1:i])
        obj_norm = normalize_name(potential_obj)
        obj = obj_lookup.get(obj_norm)
        
        if obj:
            return obj, attr
    
    return None

class AttributeDataset:
    """Dataset for attribute-based images"""
    def __init__(self, root_dir, category, mode='compound', transform=None):
        self.root_dir = root_dir
        self.category = category
        self.mode = mode
        self.transform = transform
        self.image_paths = []
        
        self.obj_lookup, self.attr_lookup, self.obj_to_attr, self.attr_to_objs = create_lookup_tables(category)
        
        category_path = osp.join(root_dir, category, mode)
        
        if osp.exists(category_path):
            self.image_paths = glob.glob(osp.join(category_path, "*.jpg"))
            self.image_paths += glob.glob(osp.join(category_path, "*.png"))
            self.image_paths += glob.glob(osp.join(category_path, "*.jpeg"))
            self.image_paths = sorted(self.image_paths)
        
        print(f"Found {len(self.image_paths)} {mode} images in {category}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        filename = osp.basename(image_path)
        
        if self.mode == 'compound':
            parsed = parse_compound_filename(filename, self.category, self.obj_lookup, self.attr_lookup)
            if parsed is None:
                print(f"Warning: Could not parse compound filename {filename}")
                return image, [], {}, image_path
            
            obj1, attr1, obj2, attr2 = parsed
            objects = [obj1, obj2]
            attributes = {obj1: attr1, obj2: attr2}
            return image, objects, attributes, image_path
        else:
            parsed = parse_single_filename(filename, self.category, self.obj_lookup, self.attr_lookup)
            if parsed is None:
                print(f"Warning: Could not parse single filename {filename}")
                return image, [], None, image_path
            
            obj, attr = parsed
            return image, [obj], attr, image_path


class DiffusionEvaluator:
    def __init__(self, args, category, mode):
        self.args = args
        self.category = category
        self.mode = mode
        self.device = device
        
        self.obj_lookup, self.attr_lookup, self.obj_to_attr, self.attr_to_objs = create_lookup_tables(category)
        
        self.all_attributes = sorted(set(ATTRIBUTES[category].values()))
        self.all_objects = sorted(set(ATTRIBUTES[category].keys()))
        
        self.correct_predictions = 0
        self.other_object_predictions = 0
        self.other_mistakes = 0
        self.total_classifications = 0
        
        if self.mode == 'single':
            self.per_object_stats = {obj: {'correct': 0, 'incorrect': 0, 'total': 0} 
                                    for obj in self.all_objects}
            self.confusion_matrix = np.zeros((len(self.all_attributes), len(self.all_attributes)), dtype=int)
            self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.all_attributes)}
        
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
        self.target_dataset = AttributeDataset(
            self.args.data_folder, 
            self.category, 
            mode=self.mode, 
            transform=transform
        )
        
    def _setup_noise(self):
        if self.args.noise_path is not None:
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        name = f"{self.category.replace(' & ', '_').replace(' ', '_')}_{self.mode}_v{self.args.version}_{self.args.n_trials}trials_"
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
            self.run_folder = osp.join(LOG_DIR, 'Attributes_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'Attributes', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def normalize_object_name(self, obj_name):
        """Normalize object name to match the canonical form in ATTRIBUTES"""
        obj_normalized = normalize_name(obj_name)
        return self.obj_lookup.get(obj_normalized, obj_name)

    def create_attribute_prompts(self, target_object, correct_attr):
        """Create prompts for dissimilar attributes only"""
        dissimilar_attrs = get_dissimilar_attributes(correct_attr, self.all_attributes, self.category)
        
        if correct_attr not in dissimilar_attrs:
            dissimilar_attrs.append(correct_attr)
        
        prompts = []
        prompt_attributes = []
        for attr in dissimilar_attrs:
            if self.mode == 'compound':
                prompt = f"A {attr.lower()} {target_object.lower()} and another object."
            else:
                prompt = f"A {attr.lower()} {target_object.lower()}."
            prompts.append(prompt)
            prompt_attributes.append(attr)
        
        print(f"\nEvaluating with dissimilar attributes for '{correct_attr}': {prompt_attributes}")
        
        return prompts, prompt_attributes

    def classify_prediction(self, predicted_attr, correct_attr, other_object_attr=None):
        """Classify the type of prediction made"""
        if predicted_attr == correct_attr:
            return 'correct'
        elif other_object_attr is not None and predicted_attr == other_object_attr:
            return 'other_object'
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

    def eval_prob_adaptive(self, unet, latent, text_embeds, scheduler, args, image, target_object, 
                        latent_size=64, all_noise=None, prompts=None, prompt_attributes=None):
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

    def perform_attribute_classification_compound(self, image, obj1, obj2, target_object, target_attr, other_object_attr):
        """Perform attribute classification for a specific object in compound image"""
        prompts, prompt_attributes = self.create_attribute_prompts(target_object, target_attr)
        
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
            self.args, img_input, target_object, self.latent_size, self.all_noise,
            prompts=prompts, prompt_attributes=prompt_attributes
        )
        
        predicted_attr = prompt_attributes[pred_idx]
        correct_attr = target_attr
        
        prediction_type = self.classify_prediction(predicted_attr, correct_attr, other_object_attr)
        
        return predicted_attr, correct_attr, prediction_type, prompts, pred_idx

    def perform_attribute_classification_single(self, image, obj, target_attr):
        """Perform attribute classification for a single object image"""
        prompts, prompt_attributes = self.create_attribute_prompts(obj, target_attr)
        print('prompts:', prompts)
        
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
            self.args, img_input, obj, self.latent_size, self.all_noise, 
            prompts=prompts, prompt_attributes=prompt_attributes
        )
        
        predicted_attr = prompt_attributes[pred_idx]
        correct_attr = target_attr

        print('obj:', obj)
        print('prompt_attributes:', prompt_attributes)
        print('correct_attr:', correct_attr)
        print('predicted_attr', predicted_attr)
        
        prediction_type = self.classify_prediction(predicted_attr, correct_attr, None)
        
        return predicted_attr, correct_attr, prediction_type, prompts, pred_idx

    def save_confusion_matrix(self):
        """Save confusion matrix visualization and CSV for single mode"""
        if self.mode != 'single':
            return
        
        df = pd.DataFrame(self.confusion_matrix, index=self.all_attributes, columns=self.all_attributes)
        csv_path = osp.join(self.run_folder, 'confusion_matrix.csv')
        df.to_csv(csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.all_attributes, yticklabels=self.all_attributes,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Attribute')
        plt.ylabel('True Attribute')
        plt.title(f'Attribute Classification Confusion Matrix - {self.category}')
        plt.tight_layout()
        
        fig_path = osp.join(self.run_folder, 'confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix visualization saved to {fig_path}")
        
        metrics_path = osp.join(self.run_folder, 'per_attribute_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Per-Attribute Metrics - {self.category}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, attr in enumerate(self.all_attributes):
                true_positives = self.confusion_matrix[i, i]
                false_positives = self.confusion_matrix[:, i].sum() - true_positives
                false_negatives = self.confusion_matrix[i, :].sum() - true_positives
                total_true = self.confusion_matrix[i, :].sum()
                
                recall = (true_positives / total_true * 100) if total_true > 0 else 0.0
                precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0.0
                f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
                
                f.write(f"{attr.upper()}:\n")
                f.write(f"  Total samples: {total_true}\n")
                f.write(f"  Correct predictions: {true_positives}\n")
                f.write(f"  Precision: {precision:.2f}%\n")
                f.write(f"  Recall: {recall:.2f}%\n")
                f.write(f"  F1-Score: {f1:.2f}\n\n")
        
        print(f"Per-attribute metrics saved to {metrics_path}")

    def save_per_object_metrics(self):
        """Save per-object metrics for single mode"""
        if self.mode != 'single':
            return
        
        metrics_path = osp.join(self.run_folder, 'per_object_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Per-Object Metrics - {self.category}\n")
            f.write("=" * 70 + "\n\n")
            
            sorted_objects = sorted(self.per_object_stats.items(), 
                                   key=lambda x: x[1]['total'], reverse=True)
            
            for obj, stats in sorted_objects:
                if stats['total'] == 0:
                    continue
                    
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                
                f.write(f"{obj.upper()}:\n")
                f.write(f"  Total images: {stats['total']}\n")
                f.write(f"  Correct predictions: {stats['correct']}\n")
                f.write(f"  Incorrect predictions: {stats['incorrect']}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n\n")
        
        print(f"Per-object metrics saved to {metrics_path}")
        
        csv_data = []
        for obj, stats in self.per_object_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total'] * 100)
                csv_data.append({
                    'Object': obj,
                    'Total': stats['total'],
                    'Correct': stats['correct'],
                    'Incorrect': stats['incorrect'],
                    'Accuracy (%)': f"{accuracy:.2f}"
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df = df.sort_values('Total', ascending=False)
            csv_path = osp.join(self.run_folder, 'per_object_metrics.csv')
            df.to_csv(csv_path, index=False)
            print(f"Per-object metrics CSV saved to {csv_path}")

    def save_results_summary(self):
        """Save a summary of attribute classification results."""
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Attribute Classification Results - {self.category} ({self.mode.title()} Mode)\n")
            f.write("=" * 70 + "\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                
                f.write(f"Attribute Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct attribute predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.mode == 'compound':
                    other_obj_acc = (self.other_object_predictions / self.total_classifications * 100)
                    other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                    f.write(f"Other object attribute predictions: {self.other_object_predictions} ({other_obj_acc:.2f}%)\n")
                    f.write(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                else:
                    other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                    f.write(f"Incorrect predictions: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                    
                    f.write(f"Per-Object Breakdown:\n")
                    f.write("-" * 70 + "\n")
                    sorted_objects = sorted(self.per_object_stats.items(), 
                                           key=lambda x: x[1]['total'], reverse=True)
                    for obj, stats in sorted_objects:
                        if stats['total'] > 0:
                            obj_acc = (stats['correct'] / stats['total'] * 100)
                            f.write(f"  {obj}: {stats['correct']}/{stats['total']} correct ({obj_acc:.2f}%), "
                                   f"{stats['incorrect']} incorrect\n")
                    f.write("\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            f.write(f"\nModel Configuration:\n")
            f.write(f"Category: {self.category}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Loss function: {self.args.loss}\n")
            f.write(f"Data type: {self.args.dtype}\n")
            f.write(f"Interpolation: {self.args.interpolation}\n")
            f.write(f"Adaptive sampling - n_samples: {self.args.n_samples}\n")
            f.write(f"Adaptive sampling - to_keep: {self.args.to_keep}\n\n")
            
            f.write(f"All possible attributes evaluated: {', '.join(self.all_attributes)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation_compound(self):
        """Run evaluation for compound images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = get_formatstr(len(self.target_dataset) * 2 - 1)
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                other_obj_acc = 100 * self.other_object_predictions / self.total_classifications
                pbar.set_description(f'{self.category} - Correct: {correct_acc:.1f}%, Other Obj: {other_obj_acc:.1f}%')
            
            image, objects, attributes, image_path = self.target_dataset[i]
            
            if len(objects) != 2 or not attributes:
                print(f"Skipping {image_path}: Invalid data")
                continue
                
            obj1, obj2 = objects
            attr1 = attributes.get(obj1)
            attr2 = attributes.get(obj2)
            
            if not attr1 or not attr2:
                print(f"Skipping {image_path}: Missing attributes")
                continue
            
            if are_attributes_similar(attr1, attr2, self.category):
                print(f"Skipping {image_path}: Attributes too similar ({attr1}, {attr2})")
                continue
            
            for obj_idx, target_object in enumerate([obj1, obj2]):
                classification_idx = i * 2 + obj_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_object':
                            self.other_object_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                target_attr = attributes[target_object]
                other_object = obj2 if target_object == obj1 else obj1
                other_object_attr = attributes[other_object]
                
                predicted_attr, correct_attr, prediction_type, prompts, pred_idx = \
                    self.perform_attribute_classification_compound(
                        image, obj1, obj2, target_object, target_attr, other_object_attr
                    )
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_object':
                    self.other_object_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_attr=predicted_attr,
                    correct_attr=correct_attr,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_object=target_object,
                    objects=[obj1, obj2],
                    attributes=attributes,
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
                pbar.set_description(f'{self.category} - Correct: {correct_acc:.2f}%')
            
            image, objects, attribute, image_path = self.target_dataset[i]
            
            if len(objects) != 1 or not attribute:
                print(f"Skipping {image_path}: Invalid data")
                continue
            
            obj = self.normalize_object_name(objects[0])
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                data = torch.load(fname)
                prediction_type = data['prediction_type']
                predicted_attr = data['predicted_attr']
                correct_attr = data['correct_attr']
                target_object = self.normalize_object_name(data['target_object'])
                
                true_idx = self.attr_to_idx[correct_attr]
                pred_idx = self.attr_to_idx[predicted_attr]
                self.confusion_matrix[true_idx, pred_idx] += 1
                
                if target_object in self.per_object_stats:
                    self.per_object_stats[target_object]['total'] += 1
                    if prediction_type == 'correct':
                        self.per_object_stats[target_object]['correct'] += 1
                    else:
                        self.per_object_stats[target_object]['incorrect'] += 1
                else:
                    print(f"Warning: Object '{target_object}' not found in per_object_stats")
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                else:
                    self.other_mistakes += 1
                self.total_classifications += 1
                
                continue
            
            predicted_attr, correct_attr, prediction_type, prompts, pred_idx = \
                self.perform_attribute_classification_single(image, obj, attribute)
            
            true_idx = self.attr_to_idx[correct_attr]
            pred_idx_cm = self.attr_to_idx[predicted_attr]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            if obj in self.per_object_stats:
                self.per_object_stats[obj]['total'] += 1
                if prediction_type == 'correct':
                    self.per_object_stats[obj]['correct'] += 1
                else:
                    self.per_object_stats[obj]['incorrect'] += 1
            else:
                print(f"Warning: Object '{obj}' not found in per_object_stats")
            
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_attr=predicted_attr,
                correct_attr=correct_attr,
                prediction_type=prediction_type,
                pred_idx=pred_idx,
                target_object=obj,
                objects=[obj],
                attribute=attribute,
                prompts=prompts,
                image_path=image_path,
                classification_idx=i
            ), fname)

    def run_evaluation(self):
        """Run evaluation based on mode"""
        if self.mode == 'compound':
            self.run_evaluation_compound()
        else:
            self.run_evaluation_single()
        
        self.save_results_summary()
        
        if self.mode == 'single':
            self.save_confusion_matrix()
            self.save_per_object_metrics()
        
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Results - {self.category} ({self.mode.title()} Mode):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.mode == 'compound':
                other_obj_acc = 100 * self.other_object_predictions / self.total_classifications
                other_mistakes_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Other object predictions: {self.other_object_predictions} ({other_obj_acc:.2f}%)")
                print(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)")
            else:
                incorrect_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Incorrect predictions: {self.other_mistakes} ({incorrect_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str,
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attributes',
                        help='Path to output folder containing category folders')
    parser.add_argument('--categories', nargs='+', 
                        default=['Part-Whole', 'Shape', 'Material & Texture', 'Size', 'Temperature'],
                        choices=['Part-Whole', 'Shape', 'Material & Texture', 'Size', 'Temperature'],
                        help='Categories to evaluate')
    parser.add_argument('--modes', nargs='+',
                        default=['single', 'compound'],
                        choices=['single', 'compound'],
                        help='Modes to evaluate (single, compound, or both)')
    
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

    parser.add_argument('--to_keep', nargs='+', default=[1], type=int)
    parser.add_argument('--n_samples', nargs='+', default=[50], type=int)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    all_results = []

    for category in args.categories:
        for mode in args.modes:
            print(f"\n{'='*70}")
            print(f"Evaluating: {category} - {mode.upper()} mode")
            print(f"{'='*70}\n")
            
            try:
                evaluator = DiffusionEvaluator(args, category, mode)
                
                if len(evaluator.target_dataset) == 0:
                    print(f"No images found for {category} - {mode}. Skipping...")
                    continue
                
                evaluator.run_evaluation()
                
                result = {
                    'category': category,
                    'mode': mode,
                    'total': evaluator.total_classifications,
                    'correct': evaluator.correct_predictions,
                    'accuracy': 100 * evaluator.correct_predictions / evaluator.total_classifications if evaluator.total_classifications > 0 else 0
                }
                
                if mode == 'compound':
                    result['other_object'] = evaluator.other_object_predictions
                    result['other_mistakes'] = evaluator.other_mistakes
                else:
                    result['incorrect'] = evaluator.other_mistakes
                
                all_results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {category} - {mode}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    if all_results:
        summary_file = osp.join(LOG_DIR, 'Attributes' if args.extra is None else f'Attributes_{args.extra}', 'overall_summary.txt')
        os.makedirs(osp.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("Overall Attribute Classification Results\n")
            f.write("=" * 70 + "\n\n")
            
            for result in all_results:
                f.write(f"\n{result['category']} - {result['mode'].upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total classifications: {result['total']}\n")
                f.write(f"Correct: {result['correct']} ({result['accuracy']:.2f}%)\n")
                
                if result['mode'] == 'compound':
                    f.write(f"Other object: {result['other_object']}\n")
                    f.write(f"Other mistakes: {result['other_mistakes']}\n")
                else:
                    f.write(f"Incorrect: {result['incorrect']}\n")
        
        print(f"\n\nOverall summary saved to {summary_file}")
        
        print("\n" + "=" * 70)
        print("OVERALL RESULTS SUMMARY")
        print("=" * 70)
        
        df_data = []
        for result in all_results:
            row = {
                'Category': result['category'],
                'Mode': result['mode'],
                'Total': result['total'],
                'Correct': result['correct'],
                'Accuracy (%)': f"{result['accuracy']:.2f}"
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        csv_file = osp.join(LOG_DIR, 'Attributes' if args.extra is None else f'Attributes_{args.extra}', 'overall_summary.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nOverall summary CSV saved to {csv_file}")


if __name__ == '__main__':
    main()