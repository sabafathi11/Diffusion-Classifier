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

# All fruits in the dataset
ALL_FRUITS = [
    'cherry', 'pomegranate', 'strawberry', 'tomato',
    'banana', 'lemon', 'corn',
    'broccoli', 'cucumber',
    'brinjal', 'plum',
    'orange', 'carrot'
]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def parse_compound_filename(filename):
    """
    Parse compound fruit filename.
    Expected format: fruit1_fruit2_number.jpg
    Example: banana_brinjal_220.jpg
    
    Returns: (fruit1, fruit2) or None if parsing fails
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    # Expected: [fruit1, fruit2, number]
    if len(parts) >= 3:
        fruit1 = parts[0]
        fruit2 = parts[1]
        if fruit1 in ALL_FRUITS and fruit2 in ALL_FRUITS:
            return fruit1, fruit2
    
    return None


def parse_single_filename(filename):
    """
    Parse single fruit filename.
    Expected format: fruit_number1_number2.jpg
    Example: banana_006_202011.jpg
    Returns: fruit or None if parsing fails
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    
    # Expected: [fruit, number1, number2] or variations
    if len(parts) >= 1:
        fruit = parts[0]
        if fruit in ALL_FRUITS:
            return fruit
    
    return None


class CABDataset:
    """Custom dataset for CAB fruit images"""
    def __init__(self, root_dir, mode='compound', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'compound':
            self.image_paths = glob.glob(osp.join(root_dir, "compound", "*.jpg"))
            self.image_paths += glob.glob(osp.join(root_dir, "compound", "*.png"))
        elif mode == 'single':
            self.image_paths = glob.glob(osp.join(root_dir, "single", "*.jpg"))
            self.image_paths += glob.glob(osp.join(root_dir, "single", "*.png"))
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
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        filename = osp.basename(image_path)
        
        if self.mode in ['compound', 'compound_unnatural']:
            parsed = parse_compound_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruit1, fruit2 = parsed
                fruits = [fruit1, fruit2]
            return image, fruits, image_path
        else:
            parsed = parse_single_filename(filename)
            if parsed is None:
                print(f"Warning: Could not parse filename {filename}")
                fruits = []
            else:
                fruits = [parsed]
            return image, fruits, image_path


class CLIPFruitEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0
        self.other_fruit_predictions = 0  # For compound mode: predicted the other fruit in image
        self.wrong_predictions = 0
        self.total_classifications = 0
        
        # Confusion matrix for all modes
        self.confusion_matrix = np.zeros((len(ALL_FRUITS), len(ALL_FRUITS)), dtype=int)
        self.fruit_to_idx = {fruit: idx for idx, fruit in enumerate(ALL_FRUITS)}
        
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
        name = f"CAB_CLIP_fruit_{self.args.mode}_v{self.args.version}"
        if self.args.img_size != 224:
            name += f'_{self.args.img_size}'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, 'CAB_CLIP_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'CAB_CLIP', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def create_fruit_prompts(self):
        """Create prompts for all fruits"""
        prompts = []
        for fruit in ALL_FRUITS:
            if self.args.prompt_template == 'simple':
                prompt = f"a {fruit}"
            elif self.args.prompt_template == 'descriptive':
                prompt = f"This is a photo of a {fruit}."
            elif self.args.prompt_template == 'natural':
                prompt = f"a photo of a {fruit}"
            else:
                prompt = f"a {fruit}"
            prompts.append(prompt)
        return prompts

    def classify_prediction(self, predicted_fruit, correct_fruit, other_fruit=None):
        """Classify the type of prediction made"""
        if predicted_fruit == correct_fruit:
            return 'correct'
        elif other_fruit is not None and predicted_fruit == other_fruit:
            return 'other_fruit'
        else:
            return 'wrong'
    
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

    def perform_fruit_classification(self, image, correct_fruit, other_fruit=None):
        """Perform fruit classification"""
        prompts = self.create_fruit_prompts()
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_fruit = ALL_FRUITS[pred_idx]
        prediction_type = self.classify_prediction(predicted_fruit, correct_fruit, other_fruit)
        
        return predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx, probs

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
            f.write(f"CAB Fruit CLIP Classification Results Summary ({self.args.mode.title()} Mode)\n")
            f.write("==========================================================\n\n")
            
            if self.total_classifications > 0:
                correct_acc = (self.correct_predictions / self.total_classifications * 100)
                wrong_acc = (self.wrong_predictions / self.total_classifications * 100)
                
                f.write(f"Fruit Classification Results:\n")
                f.write(f"Total classifications performed: {self.total_classifications}\n")
                f.write(f"Correct fruit predictions: {self.correct_predictions} ({correct_acc:.2f}%)\n")
                
                if self.args.mode in ['compound', 'compound_unnatural']:
                    other_fruit_acc = (self.other_fruit_predictions / self.total_classifications * 100)
                    f.write(f"Other fruit in image predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
                    f.write(f"Wrong predictions: {self.wrong_predictions} ({wrong_acc:.2f}%)\n\n")
                else:
                    f.write(f"Wrong predictions: {self.wrong_predictions} ({wrong_acc:.2f}%)\n\n")
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
            
            f.write(f"All possible fruits evaluated: {', '.join(ALL_FRUITS)}\n")
        
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
            
            # Classify for each fruit in the image
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx
                fname = osp.join(self.run_folder, formatstr.format(classification_idx) + '.pt')
                
                if os.path.exists(fname):
                    if self.args.load_stats:
                        data = torch.load(fname)
                        prediction_type = data['prediction_type']
                        predicted_fruit = data['predicted_fruit']
                        correct_fruit = data['correct_fruit']
                        
                        true_idx = self.fruit_to_idx[correct_fruit]
                        pred_idx = self.fruit_to_idx[predicted_fruit]
                        self.confusion_matrix[true_idx, pred_idx] += 1
                        
                        if prediction_type == 'correct':
                            self.correct_predictions += 1
                        elif prediction_type == 'other_fruit':
                            self.other_fruit_predictions += 1
                        else:
                            self.wrong_predictions += 1
                        self.total_classifications += 1
                    continue
                
                other_fruit = fruit2 if target_fruit == fruit1 else fruit1
                predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx, probs = self.perform_fruit_classification(
                    image, target_fruit, other_fruit
                )
                
                # Update confusion matrix
                true_idx = self.fruit_to_idx[correct_fruit]
                pred_idx_cm = self.fruit_to_idx[predicted_fruit]
                self.confusion_matrix[true_idx, pred_idx_cm] += 1
                
                # Update counters
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.wrong_predictions += 1
                
                self.total_classifications += 1
                
                torch.save(dict(
                    predicted_fruit=predicted_fruit,
                    correct_fruit=correct_fruit,
                    prediction_type=prediction_type,
                    pred_idx=pred_idx,
                    target_fruit=target_fruit,
                    fruits=[fruit1, fruit2],
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
                pbar.set_description(f'Correct: {correct_acc:.2f}% ({self.correct_predictions}/{self.total_classifications})')
            
            image, fruits, image_path = self.target_dataset[i]
            
            if len(fruits) != 1:
                print(f"Skipping {image_path}: Expected 1 fruit, got {len(fruits)}")
                continue
                
            fruit = fruits[0]
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                if self.args.load_stats:
                    data = torch.load(fname)
                    prediction_type = data['prediction_type']
                    predicted_fruit = data['predicted_fruit']
                    correct_fruit = data['correct_fruit']
                    
                    true_idx = self.fruit_to_idx[correct_fruit]
                    pred_idx = self.fruit_to_idx[predicted_fruit]
                    self.confusion_matrix[true_idx, pred_idx] += 1
                    
                    if prediction_type == 'correct':
                        self.correct_predictions += 1
                    else:
                        self.wrong_predictions += 1
                    self.total_classifications += 1
                continue
            
            predicted_fruit, correct_fruit, prediction_type, prompts, pred_idx, probs = self.perform_fruit_classification(
                image, fruit, None
            )
            
            # Update confusion matrix
            true_idx = self.fruit_to_idx[correct_fruit]
            pred_idx_cm = self.fruit_to_idx[predicted_fruit]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            # Update counters
            if prediction_type == 'correct':
                self.correct_predictions += 1
            else:
                self.wrong_predictions += 1
            
            self.total_classifications += 1
            
            torch.save(dict(
                predicted_fruit=predicted_fruit,
                correct_fruit=correct_fruit,
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
        if self.args.mode in ['compound', 'compound_unnatural']:
            self.run_evaluation_compound()
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
                wrong_acc = 100 * self.wrong_predictions / self.total_classifications
                print(f"Other fruit predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)")
                print(f"Wrong predictions: {self.wrong_predictions} ({wrong_acc:.2f}%)")
            else:
                wrong_acc = 100 * self.wrong_predictions / self.total_classifications
                print(f"Wrong predictions: {self.wrong_predictions} ({wrong_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--cab_folder', type=str,
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/color',
                        help='Path to color folder')
    parser.add_argument('--mode', type=str, default='single_unnatural', 
                        choices=['compound', 'single', 'compound_unnatural', 'single_unnatural'],
                        help='Mode: compound for two-fruit images, single for single-fruit images')
    
    # run args
    parser.add_argument('--version', type=str, default='1-0', help='Version identifier for logging')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (CLIP default is 224)')
    parser.add_argument('--prompt_template', type=str, default='simple', 
                        choices=['simple', 'descriptive', 'natural'],
                        help='Template for generating fruit prompts')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = CLIPFruitEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()