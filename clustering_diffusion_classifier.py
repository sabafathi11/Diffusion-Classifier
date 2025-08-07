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
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import pickle

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


class HierarchicalCluster:
    def __init__(self, class_names, max_depth=3):
        self.class_names = class_names
        self.max_depth = max_depth
        self.tree = None
        self.all_nodes = {}
        
    def build_tree(self):
        """Build hierarchical clustering tree based on class names using BERT embeddings."""
        # Create BERT embeddings from class names
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(self.class_names)
        
        # Perform hierarchical clustering
        n_classes = len(self.class_names)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage='ward',
            compute_full_tree=True
        )
        
        clustering.fit(embeddings)
        
        # Build tree structure
        self.tree = self._build_tree_structure(clustering, n_classes)
        print(f"Built hierarchical tree with {len(self.tree['children'])} top-level clusters")
        
    def _build_tree_structure(self, clustering, n_classes):
        """Build tree structure from sklearn clustering result."""
        children = clustering.children_
        
        # Create all nodes
        self.all_nodes = {}
        
        # Leaf nodes
        for i in range(n_classes):
            self.all_nodes[i] = {
                'class_indices': [i],
                'children': [],
                'is_leaf': True,
                'depth': 0
            }
        
        # Internal nodes
        for i, (left, right) in enumerate(children):
            node_id = n_classes + i
            self.all_nodes[node_id] = {
                'class_indices': self.all_nodes[left]['class_indices'] + self.all_nodes[right]['class_indices'],
                'children': [left, right],
                'is_leaf': False,
                'depth': max(self.all_nodes[left]['depth'], self.all_nodes[right]['depth']) + 1
            }
        
        # Return root node
        root_id = n_classes + len(children) - 1
        return self.all_nodes[root_id]
    
    def get_clusters_at_depth(self, depth):
        """Get all cluster nodes at a specific depth."""
        if depth == 0:
            return [self.tree]
        
        clusters = []
        queue = [(self.tree, 0)]
        
        while queue:
            node, current_depth = queue.pop(0)
            
            if current_depth == depth:
                clusters.append(node)
            elif current_depth < depth and not node['is_leaf']:
                for child_id in node['children']:
                    child_node = self._get_node_by_traversal(child_id)
                    queue.append((child_node, current_depth + 1))
        
        return clusters
    
    def _get_node_by_traversal(self, node_id):
        """Helper to get node by ID."""
        return self.all_nodes[node_id]
    
    def save_tree(self, path):
        """Save the tree structure to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'tree': self.tree, 
                'class_names': self.class_names,
                'all_nodes': self.all_nodes
            }, f)
    
    def load_tree(self, path):
        """Load the tree structure from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.tree = data['tree']
            self.class_names = data['class_names']
            self.all_nodes = data['all_nodes']


class DiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Set up models and other components
        self._setup_models()
        self._setup_dataset()
        self._setup_prompts()
        self._setup_noise()
        self._setup_run_folder()
        
        # Setup hierarchical clustering if enabled
        if self.args.use_clustering:
            self._setup_hierarchical_clustering()
        
    def _setup_models(self):
        # load pretrained models
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = get_sd_model(self.args)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        torch.backends.cudnn.benchmark = True
        
    def _setup_dataset(self):
        # set up dataset
        interpolation = INTERPOLATIONS[self.args.interpolation]
        transform = get_transform(interpolation, self.args.img_size)
        self.latent_size = self.args.img_size // 8
        self.target_dataset = get_target_dataset(self.args.dataset, train=self.args.split == 'train', transform=transform)
        
    def _setup_prompts(self):
        self.prompts_df = pd.read_csv(self.args.prompt_path)
        
        # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
        text_input = self.tokenizer(self.prompts_df.prompt.tolist(), padding="max_length",
                               max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_input.input_ids), 100):
                text_embeddings = self.text_encoder(
                    text_input.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        self.text_embeddings = torch.cat(embeddings, dim=0)
        assert len(self.text_embeddings) == len(self.prompts_df)
        
    def _setup_noise(self):
        # load noise
        if self.args.noise_path is not None:
            self.all_noise = torch.load(self.args.noise_path).to(self.device)
            print('Loaded noise from', self.args.noise_path)
        else:
            self.all_noise = None
    
    def _setup_hierarchical_clustering(self):
        """Setup hierarchical clustering based on class names extracted from classidx."""
        print("Setting up hierarchical clustering...")
        
        # Extract unique class names based on classidx
        # Create a mapping from classidx to class name (extract from prompt or use classidx directly)
        unique_classidx = sorted(self.prompts_df['classidx'].unique())
        
        # Try to extract clean class names from prompts
        # Assuming prompts are in format like "a photo of a cat" -> "cat"
        class_names = []
        classidx_to_name = {}
        
        for classidx in unique_classidx:

            class_name = self.prompts_df[self.prompts_df['classidx'] == classidx]['classname'].values[0]
            class_names.append(class_name)
            classidx_to_name[classidx] = class_name
        
        print(f"Extracted class names: {class_names}")
        self.classidx_to_name = classidx_to_name
        self.unique_classidx = unique_classidx
        
        # Initialize hierarchical clustering with class names
        self.hierarchical_cluster = HierarchicalCluster(class_names, self.args.cluster_depth)
        
        # Check if pre-computed tree exists
        cluster_cache_path = osp.join(self.run_folder, 'hierarchical_tree_labels.pkl')
        
        if osp.exists(cluster_cache_path):
            print(f"Loading pre-computed hierarchical tree from {cluster_cache_path}")
            self.hierarchical_cluster.load_tree(cluster_cache_path)
        else:
            print("Building hierarchical clustering tree...")
            self.hierarchical_cluster.build_tree()
            # Save the tree for future use
            os.makedirs(self.run_folder, exist_ok=True)
            self.hierarchical_cluster.save_tree(cluster_cache_path)
            print(f"Saved hierarchical tree to {cluster_cache_path}")
            
    def _setup_run_folder(self):
        # make run output folder
        name = f"v{self.args.version}_{self.args.n_trials}trials_"
        name += '_'.join(map(str, self.args.n_samples)) + 'samples'
        if self.args.use_clustering:
            name += f'_cluster_d{self.args.cluster_depth}_labels'
        if self.args.interpolation != 'bicubic':
            name += f'_{self.args.interpolation}'
        if self.args.loss == 'l1':
            name += '_l1'
        elif self.args.loss == 'huber':
            name += '_huber'
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

    def eval_prob_hierarchical(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
        """Evaluate probabilities using hierarchical clustering approach based on labels."""
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']
        
        max_n_samples = max(args.n_samples)
        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
        if args.dtype == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        # Start from root and traverse down the tree
        # Map from class indices (in clustering tree) to actual classidx values
        current_candidates = list(range(len(self.unique_classidx)))  # Indices in the clustering tree
        
        max_depth = min(self.args.cluster_depth, len(args.n_samples), self.hierarchical_cluster.tree['depth'])
        
        # Start from depth 1 instead of 0 (since depth 0 is just the root with all classes)
        for depth_idx, depth in enumerate(range(1, max_depth + 1)):
            if depth_idx >= len(args.n_samples):
                break
                
            n_samples_at_depth = args.n_samples[depth_idx]
            print(f"Evaluating at depth {depth} with {len(current_candidates)} candidates using {n_samples_at_depth} samples")
            
            # Get clusters at current depth that contain our candidates
            clusters_at_depth = self.hierarchical_cluster.get_clusters_at_depth(depth)
            relevant_clusters = []
            
            for cluster in clusters_at_depth:
                cluster_classes = [idx for idx in cluster['class_indices'] if idx in current_candidates]
                if cluster_classes:
                    relevant_clusters.append({
                        'class_indices': cluster_classes,
                        'representative_idx': cluster_classes[0]  # Use first class as representative
                    })
            print(f"Found {len(relevant_clusters)} relevant clusters at depth {depth}")
            
            if len(relevant_clusters) <= 1:
                print(f"Only 1 cluster found at depth {depth}, stopping hierarchical evaluation")
                break  # No more meaningful splitting
                
            # Evaluate representative classes from each cluster
            cluster_representatives = [cluster['representative_idx'] for cluster in relevant_clusters]
            
            # Setup evaluation parameters with current depth's n_samples
            start = T // n_samples_at_depth // 2
            t_to_eval = list(range(start, T, T // n_samples_at_depth))[:n_samples_at_depth]
            
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            
            for rep_class_idx in cluster_representatives:
                # Convert cluster class index to actual classidx
                actual_classidx = self.unique_classidx[rep_class_idx]
                # Find corresponding prompt indices for this classidx
                prompt_indices = self.prompts_df[self.prompts_df['classidx'] == actual_classidx].index.tolist()
                # Use the first prompt as representative
                rep_prompt_idx = prompt_indices[0]
                
                for t_idx, t in enumerate(t_to_eval):
                    ts.extend([t] * args.n_trials)
                    noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                    text_embed_idxs.extend([rep_prompt_idx] * args.n_trials)
            
            # Evaluate prediction errors
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                        text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
            
            # Compute average error for each cluster representative
            cluster_errors = {}
            error_idx = 0
            for rep_class_idx in cluster_representatives:
                rep_errors = pred_errors[error_idx:error_idx + (n_samples_at_depth * args.n_trials)]
                cluster_errors[rep_class_idx] = rep_errors.mean().item()
                error_idx += n_samples_at_depth * args.n_trials
            
            # Find best cluster (lowest error)
            best_rep_class_idx = min(cluster_errors.keys(), key=lambda x: cluster_errors[x])
            best_cluster = next(cluster for cluster in relevant_clusters if cluster['representative_idx'] == best_rep_class_idx)
            
            # Update candidates to only include classes from best cluster
            current_candidates = best_cluster['class_indices']
            print(f"Best cluster at depth {depth}: {len(current_candidates)} classes, error: {cluster_errors[best_rep_class_idx]:.4f}")
            
            # Log cluster contents for debugging
            print(f"Cluster contents at depth {depth}:")
            for i, cluster in enumerate(relevant_clusters):
                cluster_class_names = [self.classidx_to_name[self.unique_classidx[idx]] for idx in cluster['class_indices']]
                rep_idx = cluster['representative_idx']
                is_selected = (rep_idx == best_rep_class_idx)
                status = "(SELECTED)" if is_selected else ""
                
                print(f"  Cluster {i} {status}: {cluster_class_names}")
                print(f"    Representative: {self.classidx_to_name[self.unique_classidx[rep_idx]]}")
                print(f"    Error: {cluster_errors[rep_idx]:.6f}")
            print()  # Empty line for readability

        
        # Final evaluation on remaining candidates
        if len(current_candidates) > 1:
            # Use the last (highest) n_samples value for final evaluation
            final_n_samples = args.n_samples[-1] if depth_idx < len(args.n_samples) - 1 else args.n_samples[depth_idx]
            print(f"Final evaluation on {len(current_candidates)} candidates using {final_n_samples} samples")
            
            start = T // final_n_samples // 2
            t_to_eval = list(range(start, T, T // final_n_samples))[:final_n_samples]
            
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            
            # Collect all prompt indices for remaining candidate classes
            candidate_prompt_indices = []
            for candidate_class_idx in current_candidates:
                actual_classidx = self.unique_classidx[candidate_class_idx]
                prompt_indices = self.prompts_df[self.prompts_df['classidx'] == actual_classidx].index.tolist()
                candidate_prompt_indices.extend(prompt_indices)
            
            for prompt_idx in candidate_prompt_indices:
                for t_idx, t in enumerate(t_to_eval):
                    ts.extend([t] * args.n_trials)
                    noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_idx] * args.n_trials)
            
            pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                        text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
            
            # Compute average error per prompt and then per class
            class_errors = {}
            error_idx = 0
            for prompt_idx in candidate_prompt_indices:
                prompt_errors = pred_errors[error_idx:error_idx + (final_n_samples * args.n_trials)]
                classidx = self.prompts_df.iloc[prompt_idx]['classidx']
                
                if classidx not in class_errors:
                    class_errors[classidx] = []
                class_errors[classidx].append(prompt_errors.mean().item())
                error_idx += final_n_samples * args.n_trials
            
            # Average errors per class
            final_class_errors = {classidx: np.mean(errors) for classidx, errors in class_errors.items()}
            best_classidx = min(final_class_errors.keys(), key=lambda x: final_class_errors[x])
            
            # Find a prompt index for the best class
            pred_idx = self.prompts_df[self.prompts_df['classidx'] == best_classidx].index[0]
        else:
            # Only one candidate left
            actual_classidx = self.unique_classidx[current_candidates[0]]
            pred_idx = self.prompts_df[self.prompts_df['classidx'] == actual_classidx].index[0]
        
        # Create all_losses tensor for compatibility
        all_losses = torch.zeros(len(text_embeds))
        all_losses[pred_idx] = -1  # Mark the selected class
        
        return all_losses, pred_idx, {}

    def eval_prob_standard(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
        """Standard evaluation without hierarchical clustering (simplified version)."""
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']

        # Use the last (highest) n_samples value for standard evaluation
        n_samples = args.n_samples[-1] if isinstance(args.n_samples, list) else args.n_samples
        
        if all_noise is None:
            all_noise = torch.randn((n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
        if args.dtype == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        # Evaluate all classes
        start = T // n_samples // 2
        t_to_eval = list(range(start, T, T // n_samples))[:n_samples]
        
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        
        for prompt_i in range(len(text_embeds)):
            for t_idx, t in enumerate(t_to_eval):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        
        pred_errors = self.eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                     text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
        
        # Compute average error for each class
        all_losses = torch.zeros(len(text_embeds))
        error_idx = 0
        for prompt_i in range(len(text_embeds)):
            class_errors = pred_errors[error_idx:error_idx + (n_samples * args.n_trials)]
            all_losses[prompt_i] = class_errors.mean()
            error_idx += n_samples * args.n_trials
        
        pred_idx = torch.argmin(all_losses).item()
        
        return all_losses, pred_idx, {}

    def eval_error(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                   text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
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
                pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if self.args.load_stats:
                    data = torch.load(fname)
                    correct += int(data['pred'] == data['label'])
                    total += 1
                continue
            image, label = self.target_dataset[i]
            print(self.prompts_df[self.prompts_df['classidx'] == label]['classname'].values[0])
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            
            # Choose evaluation method based on clustering setting
            if self.args.use_clustering:
                _, pred_idx, pred_errors = self.eval_prob_hierarchical(
                    self.unet, x0, self.text_embeddings, self.scheduler, 
                    self.args, self.latent_size, self.all_noise
                )
            else:
                _, pred_idx, pred_errors = self.eval_prob_standard(
                    self.unet, x0, self.text_embeddings, self.scheduler, 
                    self.args, self.latent_size, self.all_noise
                )
                
            pred = self.prompts_df.classidx[pred_idx]
            torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
            if pred == label:
                correct += 1
            total += 1


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
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
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
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # hierarchical clustering args
    parser.add_argument('--use_clustering', action='store_true', help='Enable hierarchical clustering')
    parser.add_argument('--cluster_depth', type=int, default=3, help='Maximum depth for hierarchical clustering')
    parser.add_argument('--n_samples', nargs='+', type=int, default=[50, 500, 500], help='Number of samples per depth (one integer per depth level)')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = DiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()