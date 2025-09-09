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

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import DAAM components
from daam import trace, set_seed
from diffusers import StableDiffusionXLImg2ImgPipeline
from daam.heatmap import GlobalHeatMap

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


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def create_balanced_subset(dataset, samples_per_class):
    """Create a balanced subset of the dataset with specified samples per class."""
    class_to_indices = defaultdict(list)
    
    # Group indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices[label].append(idx)
    
    balanced_indices = []
    
    # Sample from each class
    for class_label, indices in class_to_indices.items():
        if len(indices) >= samples_per_class:
            # If we have enough samples, randomly select
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            # If we don't have enough samples, take all available
            selected = indices
            print(f"Warning: Class {class_label} only has {len(indices)} samples, less than requested {samples_per_class}")
        
        balanced_indices.extend(selected.tolist() if hasattr(selected, 'tolist') else selected)
    
    return sorted(balanced_indices)


class DAAMDiffusionClassifier:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking lists for plotting
        self.all_predictions = []
        self.all_labels = []
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        if self.args.prompt_path is not None:
            self._setup_prompts()
        self._setup_noise()
        self._setup_run_folder()

        
    def _setup_models(self):
        # Create diffusion pipeline for DAAM using DiffusionPipeline like in the repo example
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'  # You may want to make this configurable
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16'
        ).to(device)
        
    def _setup_dataset(self):
        # set up dataset
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        self.latent_size = self.args.img_size // 8
        self.target_dataset = get_target_dataset(self.args.dataset, train=self.args.split == 'train', transform=transform)
        
    def _setup_prompts(self):
        self.prompts_df = pd.read_csv(self.args.prompt_path)
        
        # Use classname column for DAAM processing
        self.class_names = self.prompts_df.classname.tolist()
        
        # Create mapping from class index to class name for plotting
        self.classidx_to_name = dict(zip(self.prompts_df.classidx, self.prompts_df.classname))
        
    def _setup_noise(self):
        # load noise
        if self.args.noise_path is not None:
            assert not self.args.zero_noise
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
            
    def _setup_run_folder(self):
        # make run output folder
        name = f"daam_v{self.args.version}_{self.args.n_trials}trials"
        if self.args.interpolation != 'bicubic':
            name += f'_{self.args.interpolation}'
        if self.args.img_size != 512:
            name += f'_{self.args.img_size}'
        if self.args.samples_per_class is not None:
            name += f'_{self.args.samples_per_class}spc'
        if self.args.extra is not None:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset + '_' + self.args.extra, name)
        else:
            self.run_folder = osp.join(LOG_DIR, self.args.dataset, name)
        os.makedirs(self.run_folder, exist_ok=True)
        print(f'Run folder: {self.run_folder}')

    def gaussian_center_weights_torch(self, H, W, sigma=0.35, device="cuda"):
        # Create meshgrid in torch
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        sy, sx = sigma * H, sigma * W
        w = torch.exp(-(((ys - cy) ** 2) / (2 * sy ** 2) + ((xs - cx) ** 2) / (2 * sx ** 2)))
        return w

    def gaussian_center_score_torch(self, hm, sigma=0.35, top_percent=0.1):
        # hm is a torch.Tensor on the correct device
        H, W = hm.shape
        device = hm.device
        w = self.gaussian_center_weights_torch(H, W, sigma, device)
        weighted = hm * w

        if top_percent is None:
            # Weighted mean
            return weighted.sum() / (w.sum() + 1e-8)

        flat = weighted.view(-1)
        thr = torch.quantile(flat, 1 - top_percent)
        return flat[flat >= thr].mean()

    def compute_daam_scores(self, image, class_names):
        if isinstance(image, torch.Tensor):
            image_pil = torch_transforms.ToPILImage()(image * 0.5 + 0.5)
        else:
            image_pil = image
        
        daam_scores = []
        prompt = ' '.join(class_names)

        for class_name in class_names:
            try:
                gen = set_seed(self.args.daam_seed)
                with torch.no_grad():
                    with trace(self.pipe) as tc:
                        out = self.pipe(
                            prompt=prompt,
                            image=image_pil,
                            strength=0.03,
                            num_inference_steps=self.args.daam_steps,
                            generator=gen
                        )

                        heat_map = tc.compute_global_heat_map().compute_word_heat_map(class_name)
                        hm = heat_map.heatmap  # already a torch.Tensor on GPU

                        # Normalize (1–99 percentile in torch)
                        lo = torch.quantile(hm, 0.01)
                        hi = torch.quantile(hm, 0.99)
                        hm = torch.clamp((hm - lo) / (hi - lo + 1e-8), 0, 1)

                        # Score with Gaussian center weighting
                        score = self.gaussian_center_score_torch(hm, sigma=0.15, top_percent=0.05)

                        daam_scores.append(float(score))

            except Exception as e:
                print(f"Error processing class {class_name}: {e}")
                daam_scores.append( -float('inf'))

        return torch.tensor(daam_scores)

    def eval_daam_classification(self, image):
        """
        Evaluate using DAAM heat maps for all classes at once.
        Returns DAAM scores for all classes and the predicted class index.
        """
        # Compute DAAM scores for all class names
        daam_scores = self.compute_daam_scores(image, self.class_names)

        print('\n\n\n\n\n')
        print(self.class_names)
        print(daam_scores)
        
        # Find the class with highest activation
        pred_idx = torch.argmax(daam_scores).item()
        
        return daam_scores, pred_idx

    def plot_confusion_matrix(self, save_path=None):
        """Generate and save confusion matrix plot."""
        if len(self.all_predictions) == 0 or len(self.all_labels) == 0:
            print("No predictions available for confusion matrix")
            return
        
        # Get unique class labels and their names
        unique_labels = sorted(list(set(self.all_labels + self.all_predictions)))
        class_names = [self.classidx_to_name.get(label, str(label)) for label in unique_labels]
        
        # Compute confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_predictions, labels=unique_labels)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.savefig(osp.join(self.run_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {osp.join(self.run_folder, 'confusion_matrix.png')}")
        plt.close()
        
        # Save classification report
        report = classification_report(self.all_labels, self.all_predictions, 
                                     labels=unique_labels, target_names=class_names)
        report_path = osp.join(self.run_folder, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

    def save_results_summary(self):
        """Save a comprehensive summary of results."""
        if len(self.all_predictions) == 0 or len(self.all_labels) == 0:
            print("No results to summarize")
            return
            
        summary_path = osp.join(self.run_folder, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Diffusion Classification Results Summary\n")
            f.write("=======================================\n\n")
            
            # Overall accuracy
            correct = sum(1 for p, l in zip(self.all_predictions, self.all_labels) if p == l)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Overall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct predictions: {correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            # Diffusion model configuration
            f.write(f"Model Configuration:\n")
            f.write(f"Version: {self.args.version}\n")
            f.write(f"Image size: {self.args.img_size}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
            f.write(f"Number of trials: {self.args.n_trials}\n")
            f.write(f"Data type: {self.args.dtype}\n")
            f.write(f"Interpolation: {self.args.interpolation}\n")
            f.write(f"DAAM steps: {self.args.daam_steps}\n")
            f.write(f"DAAM seed: {self.args.daam_seed}\n\n")
            
            # Per-class accuracy if we have class names
            if hasattr(self, 'classidx_to_name') and len(self.classidx_to_name) > 0:
                f.write("Per-class Results:\n")
                class_stats = {}
                for true_label, pred_label in zip(self.all_labels, self.all_predictions):
                    if true_label not in class_stats:
                        class_stats[true_label] = {'total': 0, 'correct': 0}
                    class_stats[true_label]['total'] += 1
                    if true_label == pred_label:
                        class_stats[true_label]['correct'] += 1
                
                for class_idx, stats in sorted(class_stats.items()):
                    class_name = self.classidx_to_name.get(class_idx, str(class_idx))
                    class_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    f.write(f"{class_name}: {stats['correct']}/{stats['total']} ({class_acc:.1f}%)\n")
        
        print(f"Results summary saved to {summary_path}")

    def run_evaluation(self):
        # subset of dataset to evaluate
        if self.args.subset_path is not None:
            idxs = np.load(self.args.subset_path).tolist()
        elif self.args.samples_per_class is not None:
            # Create balanced subset
            idxs = create_balanced_subset(self.target_dataset, self.args.samples_per_class)
            print(f'Created balanced subset with {len(idxs)} total samples')
        else:
            idxs = list(range(len(self.target_dataset)))
        idxs_to_eval = idxs[self.args.worker_idx::self.args.n_workers]

        formatstr = get_formatstr(len(self.target_dataset) - 1)
        correct = 0
        total = 0
        pbar = tqdm.tqdm(idxs_to_eval)
        
        for i in pbar:
            if total > 0:
                pbar.set_description(f'DAAM Acc: {100 * correct / total:.2f}%')
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['pred'] == data['label'])
                    total += 1
                    # Add to tracking lists for plotting
                    self.all_predictions.append(data['pred'])
                    self.all_labels.append(data['label'])
                continue
                
            image, label = self.target_dataset[i]

            print(self.prompts_df[self.prompts_df['classidx'] == label]['classname'].values[0])
            
            # Compute DAAM-based classification scores for all classes
            daam_scores, pred_idx = self.eval_daam_classification(image)
            
            # Get predicted class from CSV mapping
            pred = self.prompts_df.classidx[pred_idx]
                
            torch.save(dict(daam_scores=daam_scores, pred=pred, label=label), fname)
            
            # Add to tracking lists for plotting
            self.all_predictions.append(pred)
            self.all_labels.append(label)
            
            if pred == label:
                correct += 1
            total += 1

        print(f'Final DAAM Classification Accuracy: {100 * correct / total:.2f}%')
        
        # Generate plots and summary after evaluation
        self.plot_confusion_matrix()
        self.save_results_summary()


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, default=None, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--samples_per_class', type=int, default=None, help='Number of samples per class for balanced subset')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    # DAAM-specific args
    parser.add_argument('--daam_steps', type=int, default=100, help='Number of inference steps for DAAM')
    parser.add_argument('--daam_seed', type=int, default=0, help='Seed for DAAM generation')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = DAAMDiffusionClassifier(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()