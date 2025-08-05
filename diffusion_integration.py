import argparse
import os
import os.path as osp
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
import tqdm
from torch.cuda.amp import autocast, GradScaler

# Import from diffusion_classifier.py
from diffusion_classifier import DiffusionEvaluator, create_balanced_subset

# Import from learnable_templates.py
from learnable_templates import TemplateLearner, TemplateConfig, TemplateTextEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class DiffusionTemplateTrainer:
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize diffusion evaluator
        self.diffusion_evaluator = DiffusionEvaluator(args)
        
        # Get class names from prompts
        self.class_names = self.diffusion_evaluator.target_dataset.classes
        
        # Initialize template learner
        self.template_config = TemplateConfig(
            num_templates=args.num_templates,
            template_dim=args.template_dim,
            text_encoder_dim=args.text_encoder_dim,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            regularization_weight=args.regularization_weight
        )
        
        self.template_learner = TemplateLearner(self.template_config, self.class_names).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.template_learner.parameters(), 
            lr=self.template_config.learning_rate
        )
        
        # Setup dataset for training
        self._setup_training_data()
        
        # Create save directory
        self.save_dir = osp.join(args.save_dir, f"templates_{args.dataset}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.scaler = GradScaler()
        
    def _setup_training_data(self):
        """Setup training data indices"""
        if self.args.subset_path is not None:
            self.train_indices = np.load(self.args.subset_path).tolist()
        elif self.args.samples_per_class is not None:
            self.train_indices = create_balanced_subset(
                self.diffusion_evaluator.target_dataset, 
                self.args.samples_per_class
            )
        else:
            # Use first N samples for training
            dataset_size = len(self.diffusion_evaluator.target_dataset)
            train_size = min(self.args.max_train_samples, dataset_size)
            self.train_indices = list(range(train_size))
        
        print(f"Training on {len(self.train_indices)} samples")
    
    def compute_diffusion_losses_batch(self, batch_indices):
        """Compute diffusion losses for a batch of samples using eval_prob_adaptive logic"""
        batch_losses = []
        batch_labels = []
        
        for idx in batch_indices:
            image, label = self.diffusion_evaluator.target_dataset[idx]
            
            # Encode image to latent space
            with torch.no_grad():
                img_input = image.to(self.device).unsqueeze(0)
                if self.args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = self.diffusion_evaluator.vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            
            # Get current template embeddings
            template_embeddings = self.template_learner.template_encoder(self.class_names)
            num_templates, num_classes, seq_len, embed_dim = template_embeddings.shape
            
            # Compute losses for each template separately using adaptive evaluation
            sample_losses = torch.zeros(num_templates, num_classes)
            
            for t_idx in range(num_templates):
                # Get embeddings for this template across all classes
                template_class_embeddings = template_embeddings[t_idx]  # [num_classes, embed_dim]
                
                # Use eval_prob_adaptive to get losses for all classes with this template
                all_losses,_1 , _2 = self.diffusion_evaluator.eval_prob_adaptive(
                    self.diffusion_evaluator.unet,
                    x0,
                    template_class_embeddings,
                    self.diffusion_evaluator.scheduler,
                    self.args,
                    self.diffusion_evaluator.latent_size,
                    self.diffusion_evaluator.all_noise
                )
                
                sample_losses[t_idx] = all_losses
            
            batch_losses.append(sample_losses)
            batch_labels.append(label)
        
        return torch.stack(batch_losses), torch.tensor(batch_labels, device=self.device)
    
    def train_epoch(self):
        """Train templates for one epoch with AMP"""
        self.template_learner.train()
        
        # Shuffle training indices
        train_indices_shuffled = np.random.permutation(self.train_indices).tolist()
        
        total_loss = 0.0
        num_batches = 0

        for i in tqdm.trange(0, len(train_indices_shuffled), self.args.batch_size):
            batch_indices = train_indices_shuffled[i:i+self.args.batch_size]

            diffusion_losses, true_labels = self.compute_diffusion_losses_batch(batch_indices)

            self.optimizer.zero_grad()

            # Use AMP autocast for forward pass
            with autocast(dtype=torch.float16 if self.args.dtype == 'float16' else torch.float32):
                results = self.template_learner(diffusion_losses, true_labels)
                loss = results['total_loss']

            # AMP backward + optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    
    def evaluate_templates(self):
        """Evaluate current templates"""
        self.template_learner.eval()
        
        # Use a subset for evaluation
        eval_indices = self.train_indices[:min(50, len(self.train_indices))]
        
        with torch.no_grad():
            diffusion_losses, true_labels = self.compute_diffusion_losses_batch(eval_indices)
            results = self.template_learner(diffusion_losses, true_labels)
            
            # Compute accuracy for each template
            batch_size, num_templates, num_classes = diffusion_losses.shape
            template_accuracies = []
            
            for t in range(num_templates):
                predictions = torch.argmin(diffusion_losses[:, t, :], dim=1)
                accuracy = (predictions == true_labels).float().mean().item()
                template_accuracies.append(accuracy)
            
            best_template_idx = np.argmax(template_accuracies)
            best_accuracy = template_accuracies[best_template_idx]
            
            print(f"Best template {best_template_idx}: {best_accuracy:.4f} accuracy")
            print(f"Template accuracies: {template_accuracies}")
            
            return best_accuracy, template_accuracies
    
    def save_templates(self, epoch):
        """Save trained templates"""
        save_path = osp.join(self.save_dir, f"templates_epoch_{epoch}.pt")
        
        # Get template embeddings and text encoder
        template_embeddings = self.template_learner.template_encoder.template_embeddings()
        
        checkpoint = {
            'epoch': epoch,
            'template_config': self.template_config,
            'template_learner_state_dict': self.template_learner.state_dict(),
            'template_embeddings': template_embeddings.detach().cpu(),
            'class_names': self.class_names,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
        print(f"Templates saved to {save_path}")
        
        # Also save the best templates separately
        best_save_path = osp.join(self.save_dir, "best_templates.pt")
        torch.save(checkpoint, best_save_path)
    
    def train(self):
        """Main training loop"""
        print(f"Starting template training for {self.args.num_epochs} epochs")
        
        best_accuracy = 0.0
        
        for epoch in range(self.args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
            
            # Train one epoch
            avg_loss = self.train_epoch()
            print(f"Average loss: {avg_loss:.4f}")
            
            # Evaluate templates
            if (epoch + 1) % self.args.eval_interval == 0:
                accuracy, _ = self.evaluate_templates()
                
                # Save if best so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_templates(epoch + 1)
                    print(f"New best accuracy: {best_accuracy:.4f}")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % self.args.save_interval == 0:
                checkpoint_path = osp.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'template_learner_state_dict': self.template_learner.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                }, checkpoint_path)
        
        print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset args (from diffusion_classifier)
    parser.add_argument('--dataset', type=str, default='pets', 
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet', 'objectnet', 'aircraft'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--prompt_path', type=str, required=False)
    parser.add_argument('--noise_path', type=str, default=None)
    parser.add_argument('--subset_path', type=str, default=None)
    parser.add_argument('--samples_per_class', type=int, default=None)
    
    # Model args (from diffusion_classifier)
    parser.add_argument('--version', type=str, default='2-0')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512))
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'))
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'))
    parser.add_argument('--to_keep', nargs='+', type=int, default=[1])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[1])
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--template_path', type=str, default=None, help='Path to trained templates')
    
    # Template learning args
    parser.add_argument('--num_templates', type=int, default=8)
    parser.add_argument('--template_dim', type=int, default=768)
    parser.add_argument('--text_encoder_dim', type=int, default=768)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--regularization_weight', type=float, default=0.01)
    
    # Training args
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./trained_templates')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = DiffusionTemplateTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()