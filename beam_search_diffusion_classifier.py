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
from sklearn.metrics import confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
import pickle
import clip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)             # sets seed for CPU
torch.cuda.manual_seed(seed)        # sets seed for current GPU
torch.cuda.manual_seed_all(seed)    # sets seed for all GPUs (if you use multi-GPU)
np.random.seed(seed) 

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
    def __init__(self, class_names, max_depth=3, clip_model_name="ViT-B/32"):
        self.class_names = class_names
        self.max_depth = max_depth
        self.clip_model_name = clip_model_name
        self.tree = None
        self.all_nodes = {}
        self.embeddings = None  # Store embeddings for centroid calculation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def build_tree(self):
        """Build hierarchical clustering tree based on class names using CLIP embeddings."""
        # Load CLIP model
        model, _ = clip.load(self.clip_model_name, device=self.device)
        model.eval()
        
        # Create CLIP embeddings from class names
        with torch.no_grad():
            # Tokenize class names
            text_tokens = clip.tokenize(self.class_names).to(self.device)
            # Get text embeddings
            text_embeddings = model.encode_text(text_tokens)
            # Normalize embeddings (CLIP embeddings are typically normalized)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            # Store embeddings for centroid calculation
            self.embeddings = text_embeddings.cpu().numpy()
        
        # Perform hierarchical clustering
        n_classes = len(self.class_names)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage='ward',
            compute_full_tree=True
        )
        
        clustering.fit(self.embeddings)
        
        # Build tree structure with centroids
        self.tree = self._build_tree_structure(clustering, n_classes)
        print(f"Built hierarchical tree with {len(self.tree['children'])} top-level clusters using CLIP {self.clip_model_name}")
        
    def _build_tree_structure(self, clustering, n_classes):
        """Build tree structure from sklearn clustering result with centroids."""
        children = clustering.children_
        
        # Create all nodes
        self.all_nodes = {}
        
        # Leaf nodes
        for i in range(n_classes):
            centroid = self.embeddings[i]
            self.all_nodes[i] = {
                'class_indices': [i],
                'children': [],
                'is_leaf': True,
                'depth': 0,
                'centroid': centroid,
                'centroid_index': i  # For leaf nodes, centroid index is the class itself
            }
        
        # Internal nodes
        for i, (left, right) in enumerate(children):
            node_id = n_classes + i
            
            # Get class indices from children
            left_indices = self.all_nodes[left]['class_indices']
            right_indices = self.all_nodes[right]['class_indices']
            all_indices = left_indices + right_indices
            
            # Calculate centroid as mean of all embeddings in this cluster
            cluster_embeddings = self.embeddings[all_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find centroid index (closest class to the centroid)
            distances = np.linalg.norm(self.embeddings[all_indices] - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            centroid_index = all_indices[closest_idx_in_cluster]
            
            self.all_nodes[node_id] = {
                'class_indices': all_indices,
                'children': [left, right],
                'is_leaf': False,
                'depth': max(self.all_nodes[left]['depth'], self.all_nodes[right]['depth']) + 1,
                'centroid': centroid,
                'centroid_index': centroid_index
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
    
    def get_cluster_info(self, node):
        """Get human-readable information about a cluster."""
        class_names = [self.class_names[i] for i in node['class_indices']]
        centroid_class = self.class_names[node['centroid_index']]
        
        return {
            'classes': class_names,
            'centroid_class': centroid_class,
            'size': len(node['class_indices']),
            'depth': node['depth'],
            'is_leaf': node['is_leaf']
        }
    
    def save_tree(self, path):
        """Save the tree structure to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'tree': self.tree, 
                'class_names': self.class_names,
                'all_nodes': self.all_nodes,
                'clip_model_name': self.clip_model_name,
                'embeddings': self.embeddings
            }, f)
    
    def load_tree(self, path):
        """Load the tree structure from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.tree = data['tree']
            self.class_names = data['class_names']
            self.all_nodes = data['all_nodes']
            # Handle backward compatibility
            if 'clip_model_name' in data:
                self.clip_model_name = data['clip_model_name']
            else:
                self.clip_model_name = "ViT-B/32"  # Default fallback
            
            # Load embeddings if available
            if 'embeddings' in data:
                self.embeddings = data['embeddings']


class DiffusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize tracking variables
        self.all_predictions = []
        self.all_labels = []
        
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
        
        # Extract clean class names from prompts
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
        cluster_cache_path = osp.join(self.run_folder, 'hierarchical_tree_labels_clip.pkl')
        
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
            name += f'_beam_search_k{self.args.beam_width}_d{self.args.cluster_depth}_labels'
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

    def get_scaled_closeness(self, values, base=15, eps=1e-2):
        std = np.std(values)
        return int(base / (std + eps))

    def eval_prob_beam_search(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
        """Evaluate probabilities using beam search with hierarchical clustering."""
        scheduler_config = get_scheduler_config(args)
        T = scheduler_config['num_train_timesteps']
        
        max_n_samples = max(args.n_samples)
        if all_noise is None:
            all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
        if args.dtype == 'float16':
            all_noise = all_noise.half()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

        # Initialize beam with all classes
        # Each beam item is a tuple: (class_indices_list, cumulative_score)
        beam = [(list(range(len(self.unique_classidx))), 0.0)]
        
        max_depth = min(self.args.cluster_depth, self.hierarchical_cluster.tree['depth'])
        
        # Beam search through hierarchical levels
        #closeness_based_n = 100
        for depth_idx, depth in enumerate(range(1, max_depth + 1)):
            n_samples_at_depth = args.n_samples[depth_idx]
            #n_samples_at_depth = closeness_based_n
            print(f"Beam search at depth {depth} with beam width {len(beam)}, using {n_samples_at_depth} samples")
            new_beam = []
            
            for beam_candidates, beam_score in beam:
                # Get clusters at current depth that contain our candidates
                clusters_at_depth = self.hierarchical_cluster.get_clusters_at_depth(depth)
                relevant_clusters = []
                
                # inside eval_prob_beam_search, where you build relevant_clusters
                for cluster in clusters_at_depth:
                    cluster_classes = [idx for idx in cluster['class_indices'] if idx in beam_candidates]
                    if cluster_classes:
                        # recompute centroid and representative within the subset
                        subset_embs = self.hierarchical_cluster.embeddings[cluster_classes]
                        centroid = subset_embs.mean(axis=0, keepdims=True)
                        dists = np.linalg.norm(subset_embs - centroid, axis=1)
                        rep_in_subset = cluster_classes[int(np.argmin(dists))]
                        relevant_clusters.append({
                            'class_indices': cluster_classes,
                            'representative_idx': rep_in_subset
                        })

                
                if len(relevant_clusters) <= 1:
                    # No meaningful splitting, keep current candidates
                    new_beam.append((beam_candidates, beam_score))
                    continue

                # Evaluate representative classes from each cluster
                cluster_representatives = [cluster['representative_idx'] for cluster in relevant_clusters]
                
                # Setup evaluation parameters
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
                
                # Add each cluster as a new beam candidate with updated score
                for cluster in relevant_clusters:
                    rep_idx = cluster['representative_idx']
                    cluster_error = cluster_errors[rep_idx]
                    # Use negative error as score (lower error = higher score)
                    new_score = beam_score - cluster_error
                    new_beam.append((cluster['class_indices'], new_score))
            
            # Keep top-k beam candidates
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:args.beam_width]
            #scores = []
            print(f"Beam search results at depth {depth}:")
            for i, (candidates, score) in enumerate(beam):
                candidate_names = [self.classidx_to_name[self.unique_classidx[idx]] for idx in candidates]
                print(f"  Beam {i+1}: {len(candidates)} classes, score: {score:.4f}")
                #scores.append(score)
                print(f"    Classes: {candidate_names}")
            #closeness_based_n = min(self.get_scaled_closeness(scores) // len(scores), 150)

        
        # Final evaluation on all remaining beam candidates
        final_candidates = []
        for beam_candidates, _ in beam:
            final_candidates.extend(beam_candidates)
        # Remove duplicates while preserving order
        seen = set()
        final_candidates = [x for x in final_candidates if not (x in seen or seen.add(x))]
        
        if len(final_candidates) > 1:
            # Use the last (highest) n_samples value for final evaluation
            final_n_samples = args.n_samples[-1]
            #final_n_samples = closeness_based_n
            print(f"Final beam search evaluation on {len(final_candidates)} candidates using {final_n_samples} samples")
            
            start = T // final_n_samples // 2
            t_to_eval = list(range(start, T, T // final_n_samples))[:final_n_samples]
            
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            
            # Collect all prompt indices for final candidate classes
            candidate_prompt_indices = []
            for candidate_class_idx in final_candidates:
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
            actual_classidx = self.unique_classidx[final_candidates[0]]
            pred_idx = self.prompts_df[self.prompts_df['classidx'] == actual_classidx].index[0]
        
        # Create all_losses tensor for compatibility
        all_losses = torch.zeros(len(text_embeds))
        all_losses[pred_idx] = -1  # Mark the selected class
        
        return all_losses, pred_idx

    def eval_prob_standard(self, unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
        """Standard evaluation without hierarchical clustering."""
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
        
        return all_losses, pred_idx

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
            f.write("Beam Search Classification Results Summary\n")
            f.write("=========================================\n\n")
            
            # Overall accuracy
            correct = sum(1 for p, l in zip(self.all_predictions, self.all_labels) if p == l)
            total = len(self.all_predictions)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"Overall Results:\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct predictions: {correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            # Beam search configuration
            if self.args.use_clustering:
                f.write(f"Beam Search Configuration:\n")
                f.write(f"Beam width: {self.args.beam_width}\n")
                f.write(f"Maximum depth: {self.args.cluster_depth}\n")
                f.write(f"Samples per depth: {self.args.n_samples}\n\n")
            
            # Per-class accuracy if we have class names
            if hasattr(self, 'classidx_to_name'):
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
            fname = osp.join(self.run_folder, formatstr.format(i) + '.pt')
            if os.path.exists(fname):
                print('Skipping', i)
                if self.args.load_stats:
                    data = torch.load(fname)
                    pred = data['pred']
                    label = data['label']
                    
                    # Add to tracking lists
                    self.all_predictions.append(pred)
                    self.all_labels.append(label)
                    
                    correct += int(pred == label)
                    total += 1
                if total > 0:
                    pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
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
                _, pred_idx = self.eval_prob_beam_search(
                    self.unet, x0, self.text_embeddings, self.scheduler, 
                    self.args, self.latent_size, self.all_noise
                )
                pred = self.prompts_df.classidx[pred_idx]
                
                # Add to tracking lists
                self.all_predictions.append(pred)
                self.all_labels.append(label)
                
                # Save prediction
                torch.save(dict(pred=pred, label=label), fname)
            else:
                _, pred_idx = self.eval_prob_standard(
                    self.unet, x0, self.text_embeddings, self.scheduler, 
                    self.args, self.latent_size, self.all_noise
                )
                pred = self.prompts_df.classidx[pred_idx]
                
                # Add to tracking lists  
                self.all_predictions.append(pred)
                self.all_labels.append(label)
                
                torch.save(dict(pred=pred, label=label), fname)
            
            if pred == label:
                correct += 1
            total += 1
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')

        # Generate analysis plots and summaries after evaluation
        print("\nGenerating analysis results...")
        
        # Generate confusion matrix
        self.plot_confusion_matrix()
        
        # Save comprehensive results summary
        self.save_results_summary()
        
        print(f"\nFinal Results:")
        print(f"Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
        
        # Debug information
        print(f"\nDebug Info:")
        print(f"  Total predictions tracked: {len(self.all_predictions)}")
        print(f"  Total labels tracked: {len(self.all_labels)}")
        print(f"  Samples processed in this run: {total}")
        print(f"  Correct predictions: {correct}")
        print(f"  Incorrect predictions: {total - correct}")
        
        # Verify tracking accuracy
        if len(self.all_predictions) == len(self.all_labels):
            tracking_correct = sum(1 for p, l in zip(self.all_predictions, self.all_labels) if p == l)
            tracking_total = len(self.all_predictions)
            tracking_accuracy = (tracking_correct / tracking_total * 100) if tracking_total > 0 else 0
            print(f"  Tracking-based accuracy: {tracking_accuracy:.2f}% ({tracking_correct}/{tracking_total})")
        
        if self.args.use_clustering:
            print(f"\nBeam Search Configuration:")
            print(f"  Beam width: {self.args.beam_width}")
            print(f"  Maximum depth: {self.args.cluster_depth}")
            print(f"  Samples per depth: {self.args.n_samples}")


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

    # beam search clustering args
    parser.add_argument('--use_clustering', action='store_true', help='Enable beam search with hierarchical clustering')
    parser.add_argument('--cluster_depth', type=int, default=4, help='Maximum depth for hierarchical clustering')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search (number of branches to keep)')
    parser.add_argument('--n_samples', nargs='+', type=int, default=[100, 200, 200, 200], help='Number of samples per depth (one integer per depth level)')

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = DiffusionEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()