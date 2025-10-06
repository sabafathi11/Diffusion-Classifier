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
from diffusion.utils import LOG_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)             # sets seed for CPU
torch.cuda.manual_seed(seed)        # sets seed for current GPU
torch.cuda.manual_seed_all(seed)    # sets seed for all GPUs (if you use multi-GPU)
np.random.seed(seed) 
random.seed(seed)

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

# Fruit color mapping
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


class CABDataset:
    """Custom dataset for CAB fruit images"""
    def __init__(self, root_dir, mode='compound', transform=None):
        self.root_dir = root_dir
        self.mode = mode  # 'compound' or 'single'
        self.transform = transform
        
        if mode == 'compound':
            # Load compound images from fruit_combinations folder
            self.image_paths = glob.glob(osp.join(root_dir, "fruit_combinations", "*.jpg"))
        else:
            # Load single images from single_images folder
            self.image_paths = glob.glob(osp.join(root_dir, "single_images", "*.jpg"))
        
        print(f"Found {len(self.image_paths)} {mode} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract fruit names from filename
        filename = osp.basename(image_path)
        # Remove extension and number suffix
        name_part = filename.replace('.jpg', '').rsplit('_', 1)[0]
        
        if self.mode == 'compound':
            # For compound images, expect two fruits separated by underscore
            fruits = name_part.split('_')
        else:
            # For single images, expect one fruit name
            fruits = [name_part]
            
        return image, fruits, image_path


class CLIPColorEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables for results
        self.correct_predictions = 0  # Correct color predictions
        self.other_fruit_predictions = 0  # Predictions that match other fruit's color (compound mode only)
        self.other_mistakes = 0  # All other incorrect predictions
        self.total_classifications = 0  # Total number of classifications performed
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        self._setup_run_folder()
    
    def _setup_models(self):
        # Load standard CLIP model (recommended options)
        print("Loading CLIP model...")
        
        # Option 1: Use the same CLIP model as Stable Diffusion 2 (OpenCLIP ViT-H/14)
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        # Option 2: Use OpenAI's original CLIP model (smaller, faster)
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Option 3: Use OpenAI's CLIP ViT-L/14 (used in Stable Diffusion 1.x)
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Move model to device
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"CLIP model loaded on {self.device}")
        
    def _setup_dataset(self):
        # Dataset doesn't need special transforms for CLIP as processor handles it
        self.target_dataset = CABDataset(self.args.cab_folder, mode=self.args.mode, transform=None)
        
    def _setup_run_folder(self):
        # make run output folder
        name = f"CAB_CLIP_allcolors_{self.args.mode}_v{self.args.version}"
        if self.args.img_size != 224:  # CLIP default size
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
            # Different prompt templates to try
            if self.args.prompt_template == 'simple':
                prompt = f"a {color} {target_fruit}"
            elif self.args.prompt_template == 'descriptive':
                prompt = f"In this picture, the color of the {target_fruit} is {color}."
            elif self.args.prompt_template == 'natural':
                prompt = f"a {target_fruit} that is {color}"
            else:  # default
                prompt = f"a {color} {target_fruit}"
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
            # Process inputs
            inputs = self.clip_processor(
                text=prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get CLIP outputs
            outputs = self.clip_model(**inputs)
            
            # Calculate similarities
            logits_per_image = outputs.logits_per_image  # image-text similarity scores
            probs = logits_per_image.softmax(dim=1)  # convert to probabilities
            
            # Get predicted class (highest probability)
            pred_idx = probs.argmax(dim=1).item()
            
            return probs.cpu(), pred_idx

    def perform_color_classification_compound(self, image, fruit1, fruit2, target_fruit):
        """Perform color classification for a specific fruit in compound image"""
        # Create prompts for all colors
        prompts = self.create_color_prompts(target_fruit)
        
        # Perform CLIP zero-shot classification
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        # Get predicted color
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[target_fruit.lower()]
        other_fruit = fruit2 if target_fruit == fruit1 else fruit1
        other_fruit_color = FRUIT_COLORS[other_fruit.lower()]
        
        # Classify the prediction
        prediction_type = self.classify_prediction(predicted_color, correct_color, other_fruit_color)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

    def perform_color_classification_single(self, image, fruit):
        """Perform color classification for a single fruit image"""
        # Create prompts for all colors
        prompts = self.create_color_prompts(fruit)
        
        # Perform CLIP zero-shot classification
        probs, pred_idx = self.clip_zero_shot_classification(image, prompts)
        
        # Get predicted color
        predicted_color = ALL_COLORS[pred_idx]
        correct_color = FRUIT_COLORS[fruit.lower()]
        
        # Classify the prediction (no other fruit in single mode)
        prediction_type = self.classify_prediction(predicted_color, correct_color, None)
        
        return predicted_color, correct_color, prediction_type, prompts, pred_idx, probs

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
                
                if self.args.mode == 'compound':
                    f.write(f"Other fruit color predictions: {self.other_fruit_predictions} ({other_fruit_acc:.2f}%)\n")
                    f.write(f"Other mistakes: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
                else:
                    f.write(f"Incorrect predictions: {self.other_mistakes} ({other_mistakes_acc:.2f}%)\n\n")
            else:
                f.write(f"No classifications performed.\n\n")
            
            if self.args.mode == 'compound':
                f.write(f"Expected for {len(self.target_dataset)} compound images: {len(self.target_dataset) * 2} classifications\n\n")
            else:
                f.write(f"Expected for {len(self.target_dataset)} single images: {len(self.target_dataset)} classifications\n\n")
            
            # Model configuration
            f.write(f"Model Configuration:\n")
            f.write(f"Mode: {self.args.mode}\n")
            f.write(f"CLIP model: stabilityai/stable-diffusion-2-base\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Prompt template: {self.args.prompt_template}\n\n")
            
            f.write(f"All possible colors evaluated: {', '.join(ALL_COLORS)}\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation_compound(self):
        """Run evaluation for compound images"""
        idxs_to_eval = list(range(len(self.target_dataset)))
        
        formatstr = f"{{:0{len(str(len(self.target_dataset) * 2 - 1))}d}}"  # *2 because 2 classifications per image
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
            
            # Perform 2 color classifications per image (one for each fruit)
            for fruit_idx, target_fruit in enumerate([fruit1, fruit2]):
                classification_idx = i * 2 + fruit_idx  # Unique index for each classification
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
                
                # Perform color classification
                predicted_color, correct_color, prediction_type, prompts, pred_idx, probs = self.perform_color_classification_compound(
                    image, fruit1, fruit2, target_fruit
                )
                
                # Update counters
                if prediction_type == 'correct':
                    self.correct_predictions += 1
                elif prediction_type == 'other_fruit':
                    self.other_fruit_predictions += 1
                else:
                    self.other_mistakes += 1
                
                self.total_classifications += 1
                
                # Save results for this classification
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

    def run_evaluation(self):
        """Run evaluation based on mode"""
        if self.args.mode == 'compound':
            self.run_evaluation_compound()
        else:
            self.run_evaluation_single()
        
        # Generate summary
        self.save_results_summary()
        
        # Print final results
        if self.total_classifications > 0:
            correct_acc = 100 * self.correct_predictions / self.total_classifications
            print(f"\nFinal Color Classification Results ({self.args.mode.title()} Mode):")
            print(f"Total classifications: {self.total_classifications}")
            print(f"Correct color predictions: {self.correct_predictions} ({correct_acc:.2f}%)")
            
            if self.args.mode == 'compound':
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
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/CAB',
                        help='Path to CAB folder containing fruit_combinations and single_images folders')
    parser.add_argument('--mode', type=str, default='compound', choices=['compound', 'single'],
                        help='Mode: compound for two-fruit images, single for single-fruit images')
    
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