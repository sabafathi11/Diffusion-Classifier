import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pandas as pd
from diffusion.models import get_sd_model
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image

#image = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/image4.pt')
#label = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/label4.pt')
errors = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/true_label_error.pt')
timesteps = [11,31,51,71,91,111,131,151,171,191,211,231,251,271,291,311,331,351,371,391,411,431,451,471,491,511,531,551,571,591,611,631]

print('\n\n\n\n\n')


def save_pooled_image(error,timestep, path):
    os.makedirs(path, exist_ok=True)
    patch_size = 8
    channels, height, width = error.shape

    # pooled tensor: (4, 8, 8)
    pooled = torch.zeros((channels, height // patch_size, width // patch_size))

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            h_idx = i // patch_size
            w_idx = j // patch_size
            for c in range(channels):
                patch = error[c, i:i+patch_size, j:j+patch_size]
                pooled[c, h_idx, w_idx] = patch.mean()

    fig, axs = plt.subplots(1, channels, figsize=(12, 3))
    for c in range(channels):
        axs[c].imshow(pooled[c].cpu().numpy(), cmap='gray', vmin=pooled[c].min(), vmax=pooled[c].max())
        axs[c].set_title(f'Ch {c}')
        axs[c].axis('off')

    plt.tight_layout()
    plt.savefig(f"{path}/pooled_timestep{timestep}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    return pooled



for e,ts in zip(errors,timesteps):
    save_pooled_image(e,ts,f"/root/saba/diffusion-classifier/patching/exp3/TrueLabel/pooled/mean")
    
print("DONE!")