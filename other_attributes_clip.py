import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import tqdm
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from diffusion.utils import LOG_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 
random.seed(seed)

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
        {"leafy", "petaled"},
        {"feathered", "finned"},
        {"feathered", "winged"},
        {"tailed", "finned"},
        {"tailed", "legged"},
        {"feathered", "legged"},
    ],
    "Shape": [
        {"round"},
        {"square"}
    ],
    "Material & Texture": [
        {"fabric", "leather"},
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
    for split_point in range(2, len(parts) - 1):
        attr1_norm = normalize_name(parts[0])
        attr1 = attr_lookup.get(attr1_norm)
        
        if not attr1:
            continue
        
        # Try to match object1
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
        
        # Second pair
        attr2_norm = normalize_name(parts[attr2_start])
        attr2 = attr_lookup.get(attr2_norm)
        
        if not attr2:
            continue
        
        # Try to match object2
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
    for i in range(len(parts), 1, -1):
        potential_obj = '_'.join(parts[1:i])
        obj_norm = normalize_name(potential_obj)
        obj = obj_lookup.get(obj_norm)
        
        if obj:
            return obj, attr
    
    return None

class AttributeDataset:
    """Dataset for attribute-based images"""
    def __init__(self, root_dir, category, mode='compound'):
        self.root_dir = root_dir
        self.category = category
        self.mode = mode
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
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
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


class CLIPAttributeEvaluator:
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
        self._setup_run_folder()
        
    def _setup_models(self):
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        print(f"CLIP model loaded on {self.device}")
        
    def _setup_dataset(self):
        self.target_dataset = AttributeDataset(
            self.args.data_folder, 
            self.category, 
            mode=self.mode
        )
        
    def _setup_run_folder(self):
        name = f"{self.category.replace(' & ', '_').replace(' ', '_')}_{self.mode}_CLIP_v{self.args.version}"
        if self.args.img_size != 224:
            name += f'_{self.args.img_size}'
        
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, 'Attributes_CLIP_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'Attributes_CLIP', name)
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

    def clip_zero_shot_classification(self, image, prompts):
        """Perform zero-shot classification using CLIP"""
        with torch.no_grad():
            inputs = self.clip_processor(
                text=prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            pred_idx = probs.argmax(dim=1).item()
            return probs.cpu(), pred_idx

    def perform_attribute_classification_compound(self, image, obj1, obj2, target_object, target_attr, other_object_attr):
        """Perform attribute classification for a specific object in compound image"""
        prompts, prompt_attributes = self.create_attribute_prompts(target_object, target_attr)
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_attr = prompt_attributes[pred_idx]
        correct_attr = target_attr
        
        prediction_type = self.classify_prediction(predicted_attr, correct_attr, other_object_attr)
        
        return predicted_attr, correct_attr, prediction_type, prompts, pred_idx, probs

    def perform_attribute_classification_single(self, image, obj, target_attr):
        """Perform attribute classification for a single object image"""
        prompts, prompt_attributes = self.create_attribute_prompts(obj, target_attr)
        print('prompts:', prompts)
        
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_attr = prompt_attributes[pred_idx]
        correct_attr = target_attr

        print('obj:', obj)
        print('prompt_attributes:', prompt_attributes)
        print('correct_attr:', correct_attr)
        print('predicted_attr', predicted_attr)
        
        prediction_type = self.classify_prediction(predicted_attr, correct_attr, None)
        
        return predicted_attr, correct_attr, prediction_type, prompts, pred_idx, probs

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
            f.write(f"Model: CLIP (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n\n")
            
            f.write(f"All possible attributes evaluated: {', '.join(self.all_attributes)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation_compound(self):
        """Run evaluation for compound images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = f"{{:0{len(str(len(self.target_dataset) * 2 - 1))}d}}"
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
                
                predicted_attr, correct_attr, prediction_type, prompts, pred_idx, probs = \
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
                    probs=probs,
                    image_path=image_path,
                    classification_idx=classification_idx
                ), fname)

    def run_evaluation_single(self):
        """Run evaluation for single images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = f"{{:0{len(str(len(self.target_dataset) - 1))}d}}"
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
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_attr = data['predicted_attr']
                    correct_attr = data['correct_attr']
                    target_object = self.normalize_object_name(data['target_object'])
                    
                    # Update confusion matrix
                    true_idx = self.attr_to_idx[correct_attr]
                    pred_idx = self.attr_to_idx[predicted_attr]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    # Update per-object stats
                    if target_object in self.per_object_stats:
                        self.per_object_stats[target_object]['total'] += 1
                        if prediction_type == 'correct':
                            self.per_object_stats[target_object]['correct'] += 1
                        else:
                            self.per_object_stats[target_object]['incorrect'] += 1
                    else:
                        print(f"Warning: Object '{target_object}' not found in per_object_stats")
                    
                    # Update overall stats
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.other_mistakes += 1
                    self.total_classifications += 1
                
                continue
            
            # Perform new classification
            predicted_attr, correct_attr, prediction_type, prompts, pred_idx, probs = \
                self.perform_attribute_classification_single(image, obj, attribute)
            
            # Update confusion matrix
            true_idx = self.attr_to_idx[correct_attr]
            pred_idx_cm = self.attr_to_idx[predicted_attr]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            # Update per-object stats
            if obj in self.per_object_stats:
                self.per_object_stats[obj]['total'] += 1
                if prediction_type == 'correct':
                    self.per_object_stats[obj]['correct'] += 1
                else:
                    self.per_object_stats[obj]['incorrect'] += 1
            else:
                print(f"Warning: Object '{obj}' not found in per_object_stats")
            
            # Update overall stats
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            # Save results
            torch.save(dict(
                predicted_attr=predicted_attr,
                correct_attr=correct_attr,
                prediction_type=prediction_type,
                pred_idx=pred_idx,
                target_object=obj,
                objects=[obj],
                attribute=attribute,
                prompts=prompts,
                probs=probs,
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
    
    parser.add_argument('--version', type=str, default='2-0', help='Version identifier')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (CLIP default is 224)')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')

    args = parser.parse_args()

    all_results = []

    for category in args.categories:
        for mode in args.modes:
            print(f"\n{'='*70}")
            print(f"Evaluating: {category} - {mode.upper()} mode")
            print(f"{'='*70}\n")
            
            try:
                evaluator = CLIPAttributeEvaluator(args, category, mode)
                
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
        summary_file = osp.join(LOG_DIR, 'Attributes_CLIP' if args.extra is None else f'Attributes_CLIP_{args.extra}', 'overall_summary.txt')
        os.makedirs(osp.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("Overall Attribute Classification Results (CLIP)\n")
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
        print("OVERALL RESULTS SUMMARY (CLIP)")
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
        
        csv_file = osp.join(LOG_DIR, 'Attributes_CLIP' if args.extra is None else f'Attributes_CLIP_{args.extra}', 'overall_summary.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nOverall summary CSV saved to {csv_file}")


if __name__ == '__main__':
    main()