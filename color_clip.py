import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
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

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

# Fruit color mapping (natural colors)
FRUIT_COLORS = {
    'cherry': 'red',
    'pomegranate': 'red', 
    'strawberry': 'red',
    'tomato': 'red',
    'banana': 'yellow',
    'lemon': 'yellow',
    'corn': 'yellow',
    'broccoli': 'green',
    'cucumber': 'green',
    'brinjal': 'purple',
    'plum': 'purple',
    'orange': 'orange',
    'carrot': 'orange'
}

# All possible colors for evaluation
ALL_COLORS = ['yellow', 'red', 'green', 'purple', 'orange']


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def parse_compound_unnatural_filename(filename):
    """
    Parse unnatural fruit combination filename.
    Expected format: compound_fruit1_color1_fruit2_color2_number.png
    Example: compound_banana_green_brinjal_red_220.png
    
    Returns: (fruit1, color1, fruit2, color2) or None if parsing fails
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Expected: ['compound', fruit1, color1, fruit2, color2, number]
    if len(parts) >= 6 and parts[0] == 'compound':
        fruit1 = parts[1]
        color1 = parts[2]
        fruit2 = parts[3]
        color2 = parts[4]
        return fruit1, color1, fruit2, color2
    
    return None


def parse_single_unnatural_filename(filename):
    """
    Parse single unnatural fruit filename.
    Expected format: fruit_color_number1_number2.png
    Example: banana_green_006_202011.png
    Returns: (fruit, color) or None if parsing fails
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Expected: [fruit, color, number1, number2]
    if len(parts) >= 3:
        fruit = parts[0]
        color = parts[1]
        
        # Verify color is valid
        if color in ALL_COLORS:
            return fruit, color
    
    return None


class CABDataset:
    """Custom dataset for CAB fruit images"""
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
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        filename = osp.basename(image_path)
        
        if self.mode == 'compound_unnatural':
            # Parse the filename to extract fruit names and colors
            parsed = parse_compound_unnatural_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
                colors = {}
            else:
                fruit1, color1, fruit2, color2 = parsed
                fruits = [fruit1, fruit2]
                colors = {fruit1: color1, fruit2: color2}
            return image, fruits, colors, image_path
        elif self.mode == 'single_unnatural':
            # Parse the filename to extract fruit name and color
            parsed = parse_single_unnatural_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
                color = None
            else:
                fruit, color = parsed
                fruits = [fruit]
            return image, fruits, color, image_path
        else:
            # Original behavior for compound and single modes
            name_part = filename.replace('.jpg', '').rsplit('_', 1)[0]
            
            if self.mode == 'compound':
                fruits = name_part.split('_')
            else:
                fruits = [name_part]
                
            return image, fruits, image_path


class CLIPColorEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0
        self.other_fruit_predictions = 0
        self.other_mistakes = 0
        self.total_classifications = 0
        
        # Confusion matrix for single modes
        if self.args.mode == 'single' or self.args.mode == 'single_unnatural':
            self.confusion_matrix = np.zeros((len(ALL_COLORS), len(ALL_COLORS)), dtype=int)
            self.color_to_idx = {color: idx for idx, color in enumerate(ALL_COLORS)}
        
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
        self.target_dataset = CABDataset(self.args.cab_folder, mode=self.args.mode, transform=None)
        
    def _setup_run_folder(self):
        name = f"CAB_CLIP_allcolors_{self.args.mode}_v{self.args.version}"
        if self.args.img_size != 224:
            name += f'_{self.args.img_size}'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, 'CAB_CLIP_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'CAB_CLIP', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def create_color_prompts(self, target_fruit):
        """Create prompts for all 5 colors for a specific fruit"""
        prompts = []
        for color in ALL_COLORS:
            if self.args.mode in ['compound', 'compound_unnatural']:
                prompt = f"A {color.lower()} {target_fruit.lower()} and another object."
            else:
                prompt = f"A {color.lower()} {target_fruit.lower()}."
            prompts.append(prompt)
        return prompts

    def classify_prediction(self, predicted_color, correct_color, other_fruit_color=None):
        """Classify the type of prediction made"""
        if predicted_color == correct_color:
            return 'correct'
        elif other_fruit_color is not None and predicted_color == other_fruit_color:
            return 'other_fruit'
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

    def perform_color_classification_compound(self, image, fruit1, fruit2, target_fruit):
        """Perform color classification for a specific fruit in compound image"""
        prompts = self.create_color_prompts(target_fruit)
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[target_fruit.lower()]
        other_fruit = fruit2 if target_fruit == fruit1 else fruit1
        other_fruit_color = FRUIT_COLORS[other_fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, other_fruit_color)
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

    def perform_color_classification_compound_unnatural(self, image, fruit1, fruit2, target_fruit, target_color, other_fruit_color):
        """Perform color classification for a specific fruit in unnatural compound image"""
        prompts = self.create_color_prompts(target_fruit)
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = target_color
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, other_fruit_color)
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

    def perform_color_classification_single(self, image, fruit):
        """Perform color classification for a single fruit image"""
        prompts = self.create_color_prompts(fruit)
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[fruit.lower()]
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

    def perform_color_classification_single_unnatural(self, image, fruit, target_color):
        """Perform color classification for a single unnatural fruit image"""
        prompts = self.create_color_prompts(fruit)
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = target_color
        
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

    def save_confusion_matrix(self):
        """Save confusion matrix visualization and CSV for single modes"""
        if self.args.mode not in ['single', 'single_unnatural']:
            return
        
        # Save as CSV
        df = pd.DataFrame(self.confusion_matrix, index=ALL_COLORS, columns=ALL_COLORS)
        csv_path = osp.join(self.run_folder, 'confusion_matrix.csv')
        df.to_csv(csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ALL_COLORS, yticklabels=ALL_COLORS,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Color')
        plt.ylabel('True Color')
        plt.title('Color Classification Confusion Matrix')
        plt.tight_layout()
        
        fig_path = osp.join(self.run_folder, 'confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix visualization saved to {fig_path}")
        
        # Calculate and save per-class metrics
        metrics_path = osp.join(self.run_folder, 'per_class_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Per-Class Metrics\n")
            f.write("=" * 50 + "\n\n")
            
            for i, color in enumerate(ALL_COLORS):
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
                
                f.write(f"{color.upper()}:\n")
                f.write(f"  Total samples: {total_true}\n")
                f.write(f"  Correct predictions: {true_positives}\n")
                f.write(f"  Precision: {precision:.2f}%\n")
                f.write(f"  Recall: {recall:.2f}%\n")
                f.write(f"  F1-Score: {f1:.2f}\n\n")
        
        print(f"Per-class metrics saved to {metrics_path}")

    def save_results_summary(self):
        """Save a summary of color classification results."""
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"CAB Color CLIP Classification Results Summary ({self.args.mode.title()} Mode)\n")
            f.write("==========================================================\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                other_fruit_acc = (self.other_fruit_predictions / self.total_classifications * 100)
                other_mistakes_acc = (self.other_mistakes / self.total_classifications * 100)
                
                f.write(f"Color Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.args.mode in ['compound', 'compound_unnatural']:
                    f.write(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
                    f.write(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                else:
                    f.write(f"Incorrect predictions: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            if self.args.mode in ['compound', 'compound_unnatural']:
                f.write(f"Expected for {len(self.target_dataset)} compound images: {len(self.target_dataset) * 2} classifications\n\n")
            else:
                f.write(f"Expected for {len(self.target_dataset)} single images: {len(self.target_dataset)} classifications\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"Mode: {self.args.mode}\n")
            f.write(f"CLIP model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Prompt template: {self.args.prompt_template}\n\n")
            
            f.write(f"All possible colors evaluated: {', '.join(ALL_COLORS)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation_compound(self):
        """Run evaluation for compound images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        formatstr = f"{{:0{len(str(len(self.target_dataset) * 2 - 1))}d}}"
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
            if fruit1.lower() not in FRUIT_COLORS or fruit2.lower() not in FRUIT_COLORS:
                print(f"Skipping {image_path}: Unknown fruits {fruit1}, {fruit2}")
                continue
            
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {classification_idx}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                predicted_color, correct_color, prediction_type, prompts, pred_idx, probs = self.perform_color_classification_compound(
                    image, fruit1, fruit2, target_fruit
                )
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_color=predicted_color,
                    correct_color=correct_color,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=target_fruit,
                    fruits=[fruit1, fruit2],
                    prompts=prompts,
                    probs=probs,
                    image_path=image_path,
                    classification_idx=classification_idx
                ), fname)

    def run_evaluation_compound_unnatural(self):
        """Run evaluation for unnatural fruit combination images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        formatstr = f"{{:0{len(str(len(self.target_dataset) * 2 - 1))}d}}"
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.1f}%, Other Fruit: {other_fruit_acc:.1f}% ({self.total_classifications})')
            
            image, fruits, colors, image_path = self.target_dataset[i]
            
            if len(fruits) != 2:
                print(f"Skipping {image_path}: Expected 2 fruits, got {len(fruits)}")
                continue
            
            if not colors or len(colors) != 2:
                print(f"Skipping {image_path}: Could not parse colors from filename")
                continue
                
            fruit1, fruit2 = fruits
            color1 = colors.get(fruit1)
            color2 = colors.get(fruit2)
            
            if color1 not in ALL_COLORS or color2 not in ALL_COLORS:
                print(f"Skipping {image_path}: Invalid colors {color1}, {color2}")
                continue
            
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {classification_idx}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                target_color = colors[target_fruit]
                other_fruit = fruit2 if target_fruit == fruit1 else fruit1
                other_fruit_color = colors[other_fruit]
                
                predicted_color, correct_color, prediction_type, prompts, pred_idx, probs = self.perform_color_classification_compound_unnatural(
                    image, fruit1, fruit2, target_fruit, target_color, other_fruit_color
                )
                
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_color=predicted_color,
                    correct_color=correct_color,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=target_fruit,
                    target_color=target_color,
                    fruits=[fruit1, fruit2],
                    colors=colors,
                    prompts=prompts,
                    probs=probs,
                    image_path=image_path,
                    classification_idx=classification_idx
                ), fname)

    def run_evaluation_single(self):
            """Run evaluation for single images"""
            idxs_to_eval = list(range(len(self.target_dataset)))
            
            formatstr = f"{{:0{len(str(len(self.target_dataset) - 1))}d}}"  # 1 classification per image
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
                if fruit.lower() not in FRUIT_COLORS:
                    print(f"Skipping {image_path}: Unknown fruit {fruit}")
                    continue
                
                # Perform 1 color classification per image
                fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
                
                if os.path.exists(fname):
                    print(f'Skipping classification {i}')
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        else:
                            self.other_mistakes += 1
                        self.total_classifications += 1
                    continue
                
                # Perform color classification
                predicted_color, correct_color, prediction_type, prompts, pred_idx, probs = self.perform_color_classification_single(
                    image, fruit
                )
                
                # Update counters
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                # Save results for this classification
                torch.save(dict(
                    predicted_color=predicted_color,
                    correct_color=correct_color,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=fruit,
                    fruits=[fruit],
                    prompts=prompts,
                    probs=probs,
                    image_path=image_path,
                    classification_idx=i
                ), fname)
        
    def run_evaluation_single_unnatural(self):
        """Run evaluation for single unnatural images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        formatstr = f"{{:0{len(str(len(self.target_dataset) - 1))}d}}"
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                correct_acc = 100 * self.correct_predictions / self.total_classifications
                pbar.set_description(f'Correct: {correct_acc:.2f}% ({self.correct_predictions}/{self.total_classifications})')
            
            image, fruits, color, image_path = self.target_dataset[i]
            
            if len(fruits) != 1:
                print(f"Skipping {image_path}: Expected 1 fruit, got {len(fruits)}")
                continue
            
            if color is None or color not in ALL_COLORS:
                print(f"Skipping {image_path}: Invalid or missing color in filename")
                continue
                
            fruit = fruits[0]
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                print(f'Skipping classification {i}')
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_color = data['predicted_color']
                    correct_color = data['correct_color']
                    
                    true_idx = self.color_to_idx[correct_color]
                    pred_idx = self.color_to_idx[predicted_color]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.other_mistakes += 1
                    self.total_classifications += 1
                continue
            
            predicted_color, correct_color, prediction_type, prompts, pred_idx, probs = self.perform_color_classification_single_unnatural(
                image, fruit, color
            )
            print(f"fruit :{fruit}")
            print(f"predicted_color: {predicted_color}, correct_color: {correct_color}")
            
            true_idx = self.color_to_idx[correct_color]
            pred_idx_cm = self.color_to_idx[predicted_color]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.other_mistakes += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_color=predicted_color,
                correct_color=correct_color,
                prediction_type=prediction_type,
                pred_idx=pred_idx,
                target_fruit=fruit,
                fruits=[fruit],
                prompts=prompts,
                probs=probs,
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
        
        # Save confusion matrix for single modes
        if self.args.mode in ['single', 'single_unnatural']:
            self.save_confusion_matrix()
        
        # Print final results
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Color Classification Results ({self.args.mode.title()} Mode):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.args.mode in ['compound', 'compound_unnatural']:
                other_fruit_acc = 100 * self.other_fruit_predictions / self.total_classifications
                other_mistakes_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)")
                print(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)")
            else:
                incorrect_acc = 100 * self.other_mistakes / self.total_classifications
                print(f"Incorrect predictions: {self.other_mistakes} ({incorrect_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--cab_folder', type=str,
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/color',
                        help='Path to color folder')
    parser.add_argument('--mode', type=str, default='compound_unnatural', 
                        choices=['compound', 'single', 'compound_unnatural', 'single_unnatural'],
                        help='Mode: compound for two-fruit images, single for single-fruit images, ' + 
                            'compound_unnatural for unnatural color combinations, single_unnatural for single unnatural fruits')
    
    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Version identifier for logging')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (CLIP default is 224)')
    parser.add_argument('--prompt_template', type=str, default='simple', 
                        choices=['simple', 'descriptive', 'natural'],
                        help='Template for generating color prompts')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = CLIPColorEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()