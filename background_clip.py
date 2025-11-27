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


def load_imagenet_labels(csv_path):
    """Load ImageNet class labels from CSV file.
    
    Args:
        csv_path: Path to labels.csv file with format: code,class_name
        
    Returns:
        Dictionary mapping class codes to class names
    """
    df = pd.read_csv(csv_path, header=None, names=['code', 'class_name'])
    return dict(zip(df['code'], df['class_name']))


class ImageNetBDataset:
    """Dataset for ImageNet-B images"""
    def __init__(self, root_dir, intervention_types, labels_csv):
        """
        Args:
            root_dir: Root directory containing ImageNet-B data
            intervention_types: List of intervention types to include 
                               (e.g., ['color', 'Texture', 'BLiP-Caption'])
            labels_csv: Path to labels.csv file
        """
        self.root_dir = root_dir
        self.intervention_types = intervention_types
        
        # Load class labels
        self.labels_dict = load_imagenet_labels(labels_csv)
        
        # Collect all image paths and their metadata
        self.samples = []
        
        for intervention in intervention_types:
            intervention_path = osp.join(root_dir, intervention)
            if not osp.exists(intervention_path):
                print(f"Warning: Intervention path does not exist: {intervention_path}")
                continue
                
            # Get all class folders (n02119789, etc.)
            class_folders = [d for d in os.listdir(intervention_path) 
                           if osp.isdir(osp.join(intervention_path, d)) and d.startswith('n')]
            
            for class_code in class_folders:
                class_path = osp.join(intervention_path, class_code)
                
                # Get class name from labels
                class_name = self.labels_dict.get(class_code, class_code)
                
                # Get all images in this class folder
                image_paths = glob.glob(osp.join(class_path, "*.JPEG"))
                image_paths += glob.glob(osp.join(class_path, "*.jpg"))
                image_paths += glob.glob(osp.join(class_path, "*.png"))
                
                for img_path in image_paths:
                    self.samples.append({
                        'image_path': img_path,
                        'class_code': class_code,
                        'class_name': class_name,
                        'intervention': intervention
                    })
        
        print(f"Found {len(self.samples)} images across {len(intervention_types)} intervention types")
        print(f"Intervention types: {', '.join(intervention_types)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path'])
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return (image, sample['class_code'], sample['class_name'], 
                sample['intervention'], sample['image_path'])
    
    def get_all_classes(self):
        """Return list of all unique class codes and names"""
        unique_classes = {}
        for sample in self.samples:
            unique_classes[sample['class_code']] = sample['class_name']
        return unique_classes


class CLIPImageNetBEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0
        self.total_classifications = 0
        
        # Track per-intervention results
        self.intervention_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        self._setup_models()
        self._setup_dataset()
        self._setup_run_folder()
        
        # Get all unique classes for prompts
        self.all_classes = self.target_dataset.get_all_classes()
        self.class_codes = list(self.all_classes.keys())
        self.class_names = list(self.all_classes.values())
        
        print(f"Total unique classes: {len(self.all_classes)}")
        
        # Initialize overall confusion matrix
        self.confusion_matrix = np.zeros((len(self.class_codes), len(self.class_codes)), dtype=int)
        self.code_to_idx = {code: idx for idx, code in enumerate(self.class_codes)}
        
        # Initialize per-intervention confusion matrices
        self.intervention_confusion_matrices = {}
        for intervention in self.args.interventions:
            self.intervention_confusion_matrices[intervention] = np.zeros(
                (len(self.class_codes), len(self.class_codes)), dtype=int
            )
        
    def _setup_models(self):
        print(f"Loading CLIP model: {self.args.clip_model}...")
        self.clip_model = CLIPModel.from_pretrained(self.args.clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(self.args.clip_model)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        print(f"CLIP model loaded on {self.device}")
        
    def _setup_dataset(self):
        labels_csv_path = osp.join(self.args.imagenet_b_dir, 'labels.csv')
        self.target_dataset = ImageNetBDataset(
            self.args.imagenet_b_dir, 
            self.args.interventions,
            labels_csv_path
        )
        
    def _setup_run_folder(self):
        name = f"ImageNetB_CLIP_{'_'.join(self.args.interventions)}_v{self.args.version}"
        if self.args.img_size != 224:
            name += f'_{self.args.img_size}'
        if self.args.prompt_template != 'simple':
            name += f'_{self.args.prompt_template}'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, 'ImageNetB_CLIP_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, 'ImageNetB_CLIP', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def create_class_prompts(self):
        """Create prompts for all classes"""
        prompts = []
        for class_name in self.class_names:
            if self.args.prompt_template == 'simple':
                prompt = f"a photo of a {class_name}"
            elif self.args.prompt_template == 'descriptive':
                prompt = f"a high quality photo of a {class_name}"
            elif self.args.prompt_template == 'natural':
                prompt = f"a {class_name}"
            else:
                prompt = f"a photo of a {class_name}"
            prompts.append(prompt)
        return prompts

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

    def perform_classification(self, image, true_class_code):
        """Perform classification for an ImageNet-B image"""
        prompts = self.create_class_prompts()
        
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        predicted_class_code = self.class_codes[pred_idx]
        predicted_class_name = self.class_names[pred_idx]
        
        is_correct = (predicted_class_code == true_class_code)
        
        return predicted_class_code, predicted_class_name, is_correct, prompts, pred_idx, probs

    def save_confusion_matrix(self, confusion_matrix, class_codes, class_names, prefix=""):
        """Save confusion matrix visualization and CSV
        
        Args:
            confusion_matrix: The confusion matrix to save
            class_codes: List of class codes
            class_names: List of class names
            prefix: Prefix for filename (e.g., intervention name or empty for overall)
        """
        filename_prefix = f"{prefix}_" if prefix else ""
        
        # Save as CSV
        df = pd.DataFrame(confusion_matrix, 
                         index=class_codes, 
                         columns=class_codes)
        csv_path = osp.join(self.run_folder, f'{filename_prefix}confusion_matrix.csv')
        df.to_csv(csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        # Create visualization (if not too large)
        if len(class_codes) <= 50:
            plt.figure(figsize=(20, 18))
            sns.heatmap(confusion_matrix, annot=False, fmt='d', cmap='Blues',
                       xticklabels=class_codes, yticklabels=class_codes,
                       cbar_kws={'label': 'Count'})
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            title = f'{prefix} Confusion Matrix' if prefix else 'Overall Confusion Matrix'
            plt.title(f'ImageNet-B CLIP Classification {title}')
            plt.tight_layout()
            
            fig_path = osp.join(self.run_folder, f'{filename_prefix}confusion_matrix.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix visualization saved to {fig_path}")
        
        # Calculate and save per-class metrics
        metrics_path = osp.join(self.run_folder, f'{filename_prefix}per_class_metrics.txt')
        with open(metrics_path, 'w') as f:
            title_text = f"Per-Class Metrics - {prefix}" if prefix else "Per-Class Metrics - Overall"
            f.write(f"{title_text}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (code, name) in enumerate(zip(class_codes, class_names)):
                true_positives = confusion_matrix[i, i]
                false_positives = confusion_matrix[:, i].sum() - true_positives
                false_negatives = confusion_matrix[i, :].sum() - true_positives
                total_true = confusion_matrix[i, :].sum()
                
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
                
                f.write(f"{code} ({name}):\n")
                f.write(f"  Total samples: {total_true}\n")
                f.write(f"  Correct predictions: {true_positives}\n")
                f.write(f"  Precision: {precision:.2f}%\n")
                f.write(f"  Recall: {recall:.2f}%\n")
                f.write(f"  F1-Score: {f1:.2f}\n\n")
        
        print(f"Per-class metrics saved to {metrics_path}")

    def save_all_confusion_matrices(self):
        """Save overall and per-intervention confusion matrices"""
        # Save overall confusion matrix
        print("\n" + "="*80)
        print("Saving Overall Confusion Matrix")
        print("="*80)
        self.save_confusion_matrix(self.confusion_matrix, self.class_codes, self.class_names, prefix="")
        
        # Save per-intervention confusion matrices
        for intervention in self.args.interventions:
            print("\n" + "="*80)
            print(f"Saving Confusion Matrix for Intervention: {intervention}")
            print("="*80)
            self.save_confusion_matrix(
                self.intervention_confusion_matrices[intervention],
                self.class_codes,
                self.class_names,
                prefix=intervention
            )

    def save_results_summary(self):
        """Save a summary of classification results."""
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"ImageNet-B CLIP Classification Results Summary\n")
            f.write("=" * 80 + "\n\n")
            
            if self.total_classifications > 0:
                overall_acc = (self.correct_predictions / self.total_classifications * 100)
                
                f.write(f"Overall Results:\n")
                f.write(f"Total classifications: {self.total_classifications}\n")
                f.write(f"Correct predictions: {self.correct_predictions} ({overall_acc:.2f}%)\n")
                f.write(f"Incorrect predictions: {self.total_classifications - self.correct_predictions} ({100-overall_acc:.2f}%)\n\n")
                
                f.write(f"Per-Intervention Results:\n")
                f.write("-" * 80 + "\n")
                for intervention in sorted(self.intervention_results.keys()):
                    results = self.intervention_results[intervention]
                    if results['total'] > 0:
                        acc = results['correct'] / results['total'] * 100
                        f.write(f"{intervention}:\n")
                        f.write(f"  Total: {results['total']}\n")
                        f.write(f"  Correct: {results['correct']} ({acc:.2f}%)\n\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            f.write(f"\nModel Configuration:\n")
            f.write(f"CLIP Model: {self.args.clip_model}\n")
            f.write(f"Interventions: {', '.join(self.args.interventions)}\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Prompt template: {self.args.prompt_template}\n\n")
            
            f.write(f"Total classes: {len(self.all_classes)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation(self):
        """Run evaluation on ImageNet-B dataset"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = f"{{:0{len(str(len(self.target_dataset) - 1))}d}}"
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if self.total_classifications > 0:
                acc = 100 * self.correct_predictions / self.total_classifications
                pbar.set_description(f'Accuracy: {acc:.2f}% ({self.correct_predictions}/{self.total_classifications})')
            
            image, class_code, class_name, intervention, image_path = self.target_dataset[i]
            
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            
            if os.path.exists(fname):
                if self.args.load_stats:
                    data = torch.load(fname)
                    is_correct = data['is_correct']
                    intervention_type = data['intervention']
                    predicted_code = data['predicted_class_code']
                    true_code = data['true_class_code']
                    
                    # Update overall confusion matrix
                    if true_code in self.code_to_idx and predicted_code in self.code_to_idx:
                        true_idx = self.code_to_idx[true_code]
                        pred_idx = self.code_to_idx[predicted_code]
                        self.confusion_matrix[true_idx, pred_idx] += 1
                        
                        # Update per-intervention confusion matrix
                        if intervention_type in self.intervention_confusion_matrices:
                            self.intervention_confusion_matrices[intervention_type][true_idx, pred_idx] += 1
                    
                    if is_correct:
                        self.correct_predictions += 1
                    self.total_classifications += 1
                    
                    self.intervention_results[intervention_type]['total'] += 1
                    if is_correct:
                        self.intervention_results[intervention_type]['correct'] += 1
                continue
            
            predicted_code, predicted_name, is_correct, prompts, pred_idx, probs = self.perform_classification(
                image, class_code
            )
            
            # Update overall confusion matrix
            true_idx = self.code_to_idx[class_code]
            pred_idx_cm = self.code_to_idx[predicted_code]
            self.confusion_matrix[true_idx, pred_idx_cm] += 1
            
            # Update per-intervention confusion matrix
            if intervention in self.intervention_confusion_matrices:
                self.intervention_confusion_matrices[intervention][true_idx, pred_idx_cm] += 1
            
            if is_correct:
                self.correct_predictions += 1
            
            self.total_classifications += 1
            
            # Track per-intervention results
            self.intervention_results[intervention]['total'] += 1
            if is_correct:
                self.intervention_results[intervention]['correct'] += 1
            
            # Save results
            torch.save(dict(
                predicted_class_code=predicted_code,
                predicted_class_name=predicted_name,
                true_class_code=class_code,
                true_class_name=class_name,
                is_correct=is_correct,
                intervention=intervention,
                pred_idx=pred_idx,
                prompts=prompts,
                probs=probs,
                image_path=image_path,
                classification_idx=i
            ), fname)
        
        # Generate summary and save all confusion matrices
        self.save_results_summary()
        self.save_all_confusion_matrices()
        
        # Print final results
        if self.total_classifications > 0:
            overall_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal ImageNet-B CLIP Classification Results:")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct predictions: {self.correct_predictions} ({overall_acc:.2f}%)")
            print(f"\nPer-Intervention Results:")
            for intervention in sorted(self.intervention_results.keys()):
                results = self.intervention_results[intervention]
                if results['total'] > 0:
                    acc = results['correct'] / results['total'] * 100
                    print(f"  {intervention}: {results['correct']}/{results['total']} ({acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser()

    # ImageNet-B dataset args
    parser.add_argument('--imagenet_b_dir', type=str,
                        default='saba/datasets/imagenet-b-selected',
                        help='Path to ImageNet-B root directory')
    parser.add_argument('--interventions', nargs='+', 
                        default=['BLiP-Caption', 'Class-Name', 'color', 'origin', 'Texture'],
                        choices=['BLiP-Caption', 'Class-Name', 'color', 'origin', 'Texture'],
                        help='Which intervention types to evaluate')
    
    # CLIP model args
    parser.add_argument('--clip_model', type=str, 
                        default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                        help='CLIP model to use from HuggingFace')
    
    # run args
    parser.add_argument('--version', type=str, default='1-0', help='Version identifier')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (CLIP default is 224)')
    parser.add_argument('--prompt_template', type=str, default='simple', 
                        choices=['simple', 'descriptive', 'natural'],
                        help='Template for generating prompts')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = CLIPImageNetBEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()