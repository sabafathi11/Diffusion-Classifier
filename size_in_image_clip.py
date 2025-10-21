import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import tqdm
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModelForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

LOG_DIR = './data'

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

def get_transform(size=224):
    """Transform for CLIP (typically 224x224)"""
    transform = torch_transforms.Compose([
        torch_transforms.Lambda(pad_to_square),
        torch_transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
    ])
    return transform


class MultiObjectDataset:
    """Custom dataset for multi-object images with position-based subfolders."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_data = []
        
        for position in ['left',]:
            position_dir = self.root_dir / position
            if not position_dir.exists():
                print(f"Warning: {position_dir} does not exist, skipping")
                continue
            
            image_paths = sorted(list(position_dir.glob("*.png")) + list(position_dir.glob("*.jpg")))
            
            for img_path in image_paths:
                filename = img_path.stem
                objects = filename.split('_')[1:]
                pos = position
                self.image_data.append({
                    'path': img_path,
                    'objects': objects,
                    'position': pos,
                    'filename': filename
                })
        
        if len(self.image_data) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        all_objects_set = set()
        for data in self.image_data:
            all_objects_set.update(data['objects'])
        
        self.all_objects = sorted(list(all_objects_set))
        
        print(f"Found {len(self.image_data)} images total:")
        print(f"  - left: {sum(1 for d in self.image_data if d['position'] == 'left')}")
        print(f"  - middle: {sum(1 for d in self.image_data if d['position'] == 'middle')}")
        print(f"  - right: {sum(1 for d in self.image_data if d['position'] == 'right')}")
        print(f"Found {len(self.all_objects)} unique objects: {self.all_objects}")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        image = Image.open(data['path'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, data['objects'], data['position'], data['filename']


def find_biggest_object_position(objects, position):
    """Determine which object is the biggest based on the position folder."""
    if position == 'left':
        biggest_idx = 0
    elif position == 'middle':
        biggest_idx = len(objects) // 2
    elif position == 'right':
        biggest_idx = len(objects) - 1
    else:
        raise ValueError(f"Unknown position: {position}")
    
    biggest_object = objects[biggest_idx]
    other_objects = [obj for i, obj in enumerate(objects) if i != biggest_idx]
    
    return biggest_object, other_objects


def create_prompts_for_scenario(dataset, scenario_mode):
    """Create positive and negative prompts for each image."""
    prompt_data = []
    all_objects = dataset.all_objects
    
    for idx in range(len(dataset)):
        _, objects, position, filename = dataset[idx]
        
        if len(objects) < 3:
            print(f"Warning: {filename} has less than 3 objects, skipping")
            continue
        
        biggest_object, other_objects = find_biggest_object_position(objects, position)
        
        if len(other_objects) < 2:
            print(f"Warning: {filename} needs at least 3 objects total, skipping")
            continue
        
        obj2, obj3 = other_objects[0], other_objects[1]
        
        available_wrong_objects = [obj for obj in all_objects if obj not in objects]
        
        if not available_wrong_objects:
            print(f"Warning: {filename} - no available wrong objects, skipping")
            continue
        
        wrong_object = random.choice(available_wrong_objects)
        
        if scenario_mode == 1:
            positive_prompt = f"a photo of a {biggest_object}, a {obj2}, and a {obj3}"
            negative_prompt = f"a photo of a {wrong_object}, a {obj2}, and a {obj3}"
        elif scenario_mode == 2:
            positive_prompt = f"a photo of a {obj2}, a {obj3}, and a {biggest_object}"
            negative_prompt = f"a photo of a {biggest_object}, a {obj2}, and a {wrong_object}"
        else:
            raise ValueError(f"Unknown scenario mode: {scenario_mode}")
        
        prompt_data.append({
            'filename': filename,
            'position': position,
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'objects': '_'.join(objects),
            'biggest_object': biggest_object,
            'wrong_object': wrong_object
        })
    
    return pd.DataFrame(prompt_data)


class BackgroundRemover:
    """Wrapper for BiRefNet background removal."""
    
    def __init__(self, device="cuda"):
        print("Loading BiRefNet for background removal...")
        torch.set_float32_matmul_precision("high")
        
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(device)
        
        self.transform_image = torch_transforms.Compose([
            torch_transforms.Resize((1024, 1024)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.device = device
        print("BiRefNet loaded successfully")
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from a PIL image and replace with white."""
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = torch_transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        white_background = Image.new('RGB', image_size, (255, 255, 255))
        white_background.paste(image, mask=mask)
        
        return white_background


class CLIPMultiObjectEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        self.all_predictions = []
        self.all_labels = []
        self.results_details = []
        
        self._setup_models()
        self._setup_dataset()
        self._setup_run_folder()
        self._setup_prompts()
        
        if self.args.remove_background:
            self.bg_remover = BackgroundRemover(device=self.device)
        else:
            self.bg_remover = None
        
    def _setup_models(self):
        """Load CLIP model and processor"""
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        print("CLIP model loaded successfully")
        
    def _setup_dataset(self):
        transform = get_transform(size=224)  # CLIP typically uses 224x224
        self.target_dataset = MultiObjectDataset(self.args.dataset_path, transform=transform)
        
    def _setup_prompts(self):
        self.prompts_df = create_prompts_for_scenario(
            self.target_dataset, 
            self.args.scenario_mode,
        )
        
        prompts_save_path = osp.join(self.run_folder, 'prompts.csv')
        self.prompts_df.to_csv(prompts_save_path, index=False)
        print(f"Prompts saved to {prompts_save_path}")
            
    def _setup_run_folder(self):
        name = f"clip_scenario{self.args.scenario_mode}"
        if self.args.remove_background:
            name += '_nobg'
        if self.args.extra is not None:
            name += f'_{self.args.extra}'
        
        self.run_folder = osp.join(LOG_DIR, 'multi_object_clip', name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')
    
    def process_image_pil(self, image_pil):
        """Remove background from PIL image."""
        image_nobg = self.bg_remover.remove_background(image_pil)
        bg = Image.new('RGB', image_nobg.size, (255, 255, 255))
        bg.paste(image_nobg, mask=image_nobg.split()[3] if image_nobg.mode == 'RGBA' else None)
        return bg
    
    def compute_clip_similarity(self, image_pil, text_prompts):
        """
        Compute CLIP similarity between image and text prompts.
        
        Args:
            image_pil: PIL Image
            text_prompts: list of text strings
            
        Returns:
            similarities: torch.Tensor of shape (len(text_prompts),)
        """
        # Process inputs
        inputs = self.clip_processor(
            text=text_prompts,
            images=image_pil,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            
            # Get similarity scores
            logits_per_image = outputs.logits_per_image  # Shape: (1, num_texts)
            similarities = logits_per_image[0]  # Shape: (num_texts,)
        
        return similarities

    def save_results_summary(self):
        """Save comprehensive summary of results."""
        if len(self.all_predictions) == 0:
            print("No results to summarize")
            return
            
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CLIP Multi-Object Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            correct = sum(1 for p in self.all_predictions if p == 0)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Scenario Mode: {self.args.scenario_mode}\n")
            if self.args.scenario_mode == 1:
                f.write("  - Positive: biggest object first\n")
                f.write("  - Negative: wrong object first\n")
            else:
                f.write("  - Positive: biggest object in middle\n")
                f.write("  - Negative: wrong object first\n")
            
            f.write(f"\nOverall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct (chose positive prompt): {correct}\n")
            f.write(f"Incorrect (chose negative prompt): {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            f.write("Results by Position:\n")
            for position in ['left', 'middle', 'right']:
                position_results = [r for r in self.results_details if r.get('position') == position]
                if position_results:
                    pos_correct = sum(1 for r in position_results if r['chose_positive'])
                    pos_total = len(position_results)
                    pos_acc = (pos_correct / pos_total * 100) if pos_total > 0 else 0
                    f.write(f"  {position}: {pos_correct}/{pos_total} ({pos_acc:.2f}%)\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"Model: CLIP-ViT-H-14-laion2B-s32B-b79K\n")
            f.write(f"Background removal: {self.args.remove_background}\n")
        
        print(f"Results summary saved to {summary_path}")
        
        results_df = pd.DataFrame(self.results_details)
        results_csv_path = osp.join(self.run_folder, 'detailed_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to {results_csv_path}")

    def run_evaluation(self):
        correct = 0
        total = 0
        pbar = tqdm.tqdm(range(len(self.target_dataset)))
        
        for i in pbar:
            if total > 0:
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            
            fname = osp.join(self.run_folder, f'result_{i:04d}.pt')
            if os.path.exists(fname) and self.args.load_stats:
                data = torch.load(fname)
                correct += int(data['chose_positive'])
                total += 1
                self.all_predictions.append(0 if data['chose_positive'] else 1)
                self.all_labels.append(0)
                self.results_details.append(data['result_detail'])
                continue
            
            image_pil, objects, position, filename = self.target_dataset[i]
            
            # Remove background if requested
            if self.args.remove_background:
                image_pil = self.process_image_pil(image_pil)
            
            # Get prompts for this image
            positive_prompt = self.prompts_df.iloc[i]['positive_prompt']
            negative_prompt = self.prompts_df.iloc[i]['negative_prompt']
            text_prompts = [positive_prompt, negative_prompt]
            
            # Compute CLIP similarities
            similarities = self.compute_clip_similarity(image_pil, text_prompts)
            
            # pred_idx is 0 for positive, 1 for negative
            pred_idx = similarities.argmax().item()
            chose_positive = (pred_idx == 0)
            
            result_detail = {
                'filename': filename,
                'position': position,
                'objects': '_'.join(objects),
                'chose_positive': chose_positive,
                'positive_similarity': similarities[0].item(),
                'negative_similarity': similarities[1].item(),
                'similarity_diff': (similarities[0] - similarities[1]).item(),
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt
            }
            self.results_details.append(result_detail)
            
            torch.save({
                'similarities': similarities.cpu(),
                'chose_positive': chose_positive,
                'filename': filename,
                'position': position,
                'objects': objects,
                'result_detail': result_detail
            }, fname)
            
            if chose_positive:
                correct += 1
            total += 1
            
            self.all_predictions.append(0 if chose_positive else 1)
            self.all_labels.append(0)
        
        self.save_results_summary()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/comco/3',
                        help='Path to dataset directory')
    parser.add_argument('--scenario_mode', type=int, default=2, choices=[1, 2],
                        help='Scenario mode for prompt generation')
    parser.add_argument('--remove_background', action='store_true', default=False, 
                        help='Remove background using BiRefNet')
    parser.add_argument('--extra', type=str, default=None, 
                        help='Extra string for run folder name')
    parser.add_argument('--load_stats', action='store_true', 
                        help='Load saved stats to compute accuracy')

    args = parser.parse_args()

    evaluator = CLIPMultiObjectEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()