import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class TemplateConfig:
    """Configuration for learnable templates"""
    num_templates: int = 4 # Number of learnable templates
    template_dim: int = 1024  # Dimension of learned template embeddings
    text_encoder_dim: int = 1024  # CLIP text encoder dimension
    max_length: int = 77  # Max sequence length for CLIP
    learning_rate: float = 1e-3
    temperature: float = 1.0  # For Gumbel softmax
    regularization_weight: float = 0.01


class LearnableTemplateEmbeddings(nn.Module):
    """
    Learnable template embeddings that generate different tokens for each sequence position
    """
    def __init__(self, config: TemplateConfig):
        super().__init__()
        self.config = config
        
        # Learnable template embeddings for each sequence position
        self.template_embeddings = nn.Parameter(
            torch.randn(config.num_templates, config.max_length, config.template_dim)
        )
        
        # Projection layer to convert template embeddings to text encoder space
        self.template_projector = nn.Sequential(
            nn.Linear(config.template_dim, config.text_encoder_dim),
            nn.ReLU(),
            nn.Linear(config.text_encoder_dim, config.text_encoder_dim),
            nn.LayerNorm(config.text_encoder_dim)
        )
        
        # Initialize embeddings with Xavier initialization
        nn.init.xavier_uniform_(self.template_embeddings)
        
    def forward(self, template_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to get template embeddings
        
        Args:
            template_indices: If provided, select specific templates. Otherwise return all.
            
        Returns:
            Template embeddings projected to text encoder space: [num_templates, seq_len, embed_dim]
        """
        if template_indices is not None:
            selected_embeddings = self.template_embeddings[template_indices]
        else:
            selected_embeddings = self.template_embeddings
            
        # Project each sequence position
        batch_size, seq_len, template_dim = selected_embeddings.shape
        flat_embeddings = selected_embeddings.reshape(-1, template_dim)
        projected_flat = self.template_projector(flat_embeddings)
        projected = projected_flat.reshape(batch_size, seq_len, self.config.text_encoder_dim)
        
        return projected


class TemplateTextEncoder(nn.Module):
    """
    Combines learnable templates with class labels to create text embeddings
    """
    def __init__(self, config: TemplateConfig, model_name: str = "stabilityai/stable-diffusion-2"):
        super().__init__()
        config = TemplateConfig()
        self.config = config

        # Load tokenizer and text encoder from SD v2 model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        
        # Freeze text encoder weights initially
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Learnable template embeddings
        self.template_embeddings = LearnableTemplateEmbeddings(config)
        
        # Template combination layer
        self.template_combiner = nn.MultiheadAttention(
            embed_dim=config.text_encoder_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Class name embedding cache
        self.class_embeddings_cache = {}
        
    def encode_class_names(self, class_names: List[str]) -> torch.Tensor:
        """
        Encode class names using CLIP text encoder
        """
        # Create cache key
        cache_key = tuple(sorted(class_names))
        
        if cache_key in self.class_embeddings_cache:
            return self.class_embeddings_cache[cache_key]
            
        # Tokenize class names
        tokens = self.tokenizer(
            class_names,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            device = next(self.text_encoder.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = self.text_encoder(**tokens)
            class_embeddings = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            
        self.class_embeddings_cache[cache_key] = class_embeddings
        return class_embeddings
    
    def combine_template_and_class(
        self, 
        template_embeddings: torch.Tensor, 
        class_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine template and class embeddings using attention
        
        Args:
            template_embeddings: [num_templates, seq_len, embed_dim]
            class_embeddings: [num_classes, embed_dim]
            
        Returns:
            Combined embeddings: [num_templates, num_classes, seq_len, embed_dim]  
        """
        num_templates, seq_len, embed_dim = template_embeddings.shape
        num_classes = class_embeddings.shape[0]
        
        # Expand class embeddings to sequence length
        class_embeddings_seq = class_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Expand dimensions for combination
        templates_expanded = template_embeddings.unsqueeze(1).expand(-1, num_classes, -1, -1)
        classes_expanded = class_embeddings_seq.unsqueeze(0).expand(num_templates, -1, -1, -1)
        
        # Combine by addition or concatenation - here using addition
        combined = templates_expanded + classes_expanded
        
        return combined
    def forward(self, class_names: List[str], template_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate text embeddings for all template-class combinations
        
        Args:
            class_names: List of class names
            template_indices: Optional specific template indices to use
            
        Returns:
            Combined embeddings: [num_templates, num_classes, seq_len, embed_dim]
        """
        # Get template embeddings
        template_embeddings = self.template_embeddings(template_indices)
        
        # Get class embeddings
        class_embeddings = self.encode_class_names(class_names)
        # Move to same device
        class_embeddings = class_embeddings.to(template_embeddings.device)
        
        # Combine template and class embeddings
        combined_embeddings = self.combine_template_and_class(
            template_embeddings, class_embeddings
        )
        
        return combined_embeddings


class TemplateLearner(nn.Module):
    """
    Main module for learning optimal templates for diffusion classification
    """
    def __init__(self, config: TemplateConfig, class_names: List[str]):
        super().__init__()
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Template text encoder
        self.template_encoder = TemplateTextEncoder(config)
        
        # Template selection network (for dynamic template selection)
        self.template_selector = nn.Sequential(
            nn.Linear(config.text_encoder_dim, config.num_templates * 2),
            nn.ReLU(),
            nn.Linear(config.num_templates * 2, config.num_templates),
            nn.Softmax(dim=-1)
        )
        
        # Diversity regularization
        self.diversity_loss_fn = nn.MSELoss()
        
    def compute_diversity_loss(self, template_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encourage diversity among template embeddings
        """
        # Compute pairwise similarities
        # template_embeddings shape: (num_templates, num_classes, seq_len, embed_dim)
        flattened = template_embeddings.view(template_embeddings.size(0), -1)  # (num_templates, num_classes * seq_len * embed_dim)
        norm_embeddings = F.normalize(flattened, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # We want off-diagonal elements to be small (diverse templates)
        mask = ~torch.eye(template_embeddings.size(0), dtype=torch.bool, device=template_embeddings.device)
        diversity_loss = similarity_matrix[mask].pow(2).mean()
        
        return diversity_loss
    
    def select_best_template(
        self, 
        losses_per_template: torch.Tensor, 
        use_gumbel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best template based on losses
        
        Args:
            losses_per_template: [num_templates] tensor of losses
            use_gumbel: Whether to use Gumbel softmax for differentiability
            
        Returns:
            Selected template logits and hard selection
        """
        # Convert losses to logits (lower loss = higher probability)
        logits = -losses_per_template / self.config.temperature
        
        if use_gumbel and self.training:
            # Gumbel softmax for differentiable discrete selection
            selection_soft = F.gumbel_softmax(logits, tau=self.config.temperature, hard=False)
            selection_hard = F.gumbel_softmax(logits, tau=self.config.temperature, hard=True)
            
            # Straight-through estimator
            selection = selection_hard - selection_soft.detach() + selection_soft
        else:
            # Hard selection during evaluation
            selection = torch.zeros_like(logits)
            best_idx = torch.argmin(losses_per_template)
            selection[best_idx] = 1.0
            
        return logits, selection
    
    def compute_classification_loss(
        self, 
        diffusion_losses: torch.Tensor, 
        true_labels: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute differentiable classification loss from diffusion losses
        
        Args:
            diffusion_losses: [batch_size, num_templates, num_classes]
            true_labels: [batch_size]
            temperature: Temperature for softmax (lower = more confident)
            
        Returns:
            Classification loss per template
        """
        batch_size, num_templates, num_classes = diffusion_losses.shape
        
        # Convert losses to logits (lower loss = higher probability)
        # Negative because we want lower losses to have higher probabilities
        logits = -diffusion_losses / temperature  # [batch_size, num_templates, num_classes]
        
        # Compute cross-entropy loss for each template
        template_losses = []
        for t in range(num_templates):
            template_logits = logits[:, t, :]  # [batch_size, num_classes]
            device = 'cuda' if template_logits.is_cuda else 'cpu'
            template_logits = template_logits.to(device)
            true_labels = true_labels.to(device)
            loss = F.cross_entropy(template_logits, true_labels)
            template_losses.append(loss)
        
        return torch.stack(template_losses)  # [num_templates]
    
    def forward(
        self, 
        diffusion_losses: torch.Tensor, 
        true_labels: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for template learning
        
        Args:
            diffusion_losses: [batch_size, num_templates, num_classes] 
            true_labels: [batch_size]
            return_embeddings: Whether to return template embeddings
            
        Returns:
            Dictionary with losses and optionally embeddings
        """
        # Get template embeddings
        template_embeddings = self.template_encoder.template_embeddings()
        
        # Compute classification loss for each template
        classification_losses = self.compute_classification_loss(diffusion_losses, true_labels)
        
        # Select best template
        template_logits, template_selection = self.select_best_template(classification_losses)
        
        # Compute final classification loss (weighted by template selection)
        final_classification_loss = torch.sum(template_selection * classification_losses)
        
        # Compute diversity loss
        diversity_loss = self.compute_diversity_loss(template_embeddings)
        
        # Total loss
        total_loss = (
            final_classification_loss + 
            self.config.regularization_weight * diversity_loss
        )
        
        results = {
            'total_loss': total_loss,
            'classification_loss': final_classification_loss,
            'diversity_loss': diversity_loss,
            'template_logits': template_logits,
            'template_selection': template_selection,
            'classification_losses_per_template': classification_losses
        }
        
        if return_embeddings:
            results['template_embeddings'] = template_embeddings
            
        return results

