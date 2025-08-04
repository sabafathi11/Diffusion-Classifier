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


def save_latent_channels_error(error,timestep,path):
    os.makedirs(path, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axs[i].imshow(error[i].cpu().numpy(), cmap='gray')
        axs[i].set_title(f'Channel {i}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.title(f"timestep {timestep}")
    plt.savefig(f"{path}/latent_channel_error.png")
    plt.close()

def save_latent_channels_patches_error(error,path):
    os.makedirs(path, exist_ok=True)
    patch_size = 8
    channels, height, width = error.shape
    patch_id = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            fig, axs = plt.subplots(1, channels, figsize=(12, 3))
            for c in range(channels):
                patch = error[c, i:i+patch_size, j:j+patch_size]
                axs[c].imshow(patch.cpu().numpy(), cmap='gray', vmin=patch.min(), vmax=patch.max())
                axs[c].set_title(f'Ch {c}')
                axs[c].axis('off')

            plt.tight_layout()
            plt.savefig(f"{path}/patch_{patch_id:03d}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            patch_id += 1


#save_image(image, '/root/saba/diffusion-classifier/patching/exp3/image.png')
#print(label)
for e,ts in zip(errors,timesteps):
    save_latent_channels_error(e,ts,f"/root/saba/diffusion-classifier/patching/exp3/TrueLabel/timestep{ts}")
    save_latent_channels_patches_error(e,f"/root/saba/diffusion-classifier/patching/exp3/TrueLabel/timestep{ts}/patches")
    
print("DONE!")