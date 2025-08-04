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
    num_templates: int = 8
    template_dim: int = 1024  # Dimension of learned template embeddings
    text_encoder_dim: int = 1024  # CLIP text encoder dimension
    max_length: int = 77  # Max sequence length for CLIP
    learning_rate: float = 1e-3
    temperature: float = 1.0  # For Gumbel softmax
    regularization_weight: float = 0.01


class LearnableTemplateEmbeddings(nn.Module):
    """
    Learnable template embeddings that can be converted to text prompts
    """
    def __init__(self, config: TemplateConfig):
        super().__init__()
        self.config = config
        
        # Learnable template embeddings
        self.template_embeddings = nn.Parameter(
            torch.randn(config.num_templates, config.template_dim)
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
            Template embeddings projected to text encoder space
        """
        if template_indices is not None:
            selected_embeddings = self.template_embeddings[template_indices]
        else:
            selected_embeddings = self.template_embeddings
            
        return self.template_projector(selected_embeddings)


import torch

# Instantiate config and model
config = TemplateConfig()
model = LearnableTemplateEmbeddings()

# Create random indices to select templates (e.g., batch of 4 with 3 templates each)
batch_size = 4
num_templates_per_sample = 3

# Random integers between 0 and num_templates-1
template_indices = torch.randint(low=0, high=config.num_templates, size=(batch_size, num_templates_per_sample))

# Forward pass
output_embeddings = model(template_indices)

print("Input indices shape:", template_indices.shape)  # (4, 3)
print("Output embeddings shape:", output_embeddings.shape)  # Expected (4, 3, 1024)
