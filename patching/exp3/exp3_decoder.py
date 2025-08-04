import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pandas as pd
from diffusion.models import get_sd_model
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image

image = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/image4.pt')
label = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/label4.pt')
errors = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/error4.pt')
timesteps = torch.load('/root/saba/diffusion-classifier/patching/exp3/samples/timestep4.pt')

print('\n\n\n\n\n')

import argparse

def get_args():
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
    parser.add_argument('--noise_path', type=str, default=None)
    parser.add_argument('--subset_path', type=str, default=None)
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'))
    parser.add_argument('--interpolation', type=str, default='bicubic')
    parser.add_argument('--extra', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--worker_idx', type=int, default=0)
    parser.add_argument('--load_stats', action='store_true')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'))

    # adaptive args
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    # simulate CLI arguments:
    arg_list = [
        "--dataset", "cifar10",
        "--split", "test",
        "--n_trials", "1",
        "--to_keep", "5", "1",
        "--n_samples", "50", "500",
        "--loss", "l1",
        "--prompt_path", "prompts/cifar10_prompts.csv"
    ]

    return parser.parse_args(arg_list)

args = get_args()
vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
vae = vae.to("cuda")

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
    with torch.no_grad():
        decoded_error = vae.decode(error.unsqueeze(dim=0)).sample
    image = decoded_error[0].permute(1,2,0)
    img_np = image.detach().cpu().numpy().astype("float32")
    img_np = img_np.clip(0, 1)  # Optional but safe
    plt.imshow(img_np)
    plt.axis('off')
    plt.tight_layout()
    plt.title(f"timestep {timestep}")
    plt.savefig(f"{path}/decoded_error.png")
    plt.close()


for e,ts in zip(errors,timesteps):
    save_latent_channels_error(e,ts,f"/root/saba/diffusion-classifier/patching/exp3/decode/timestep{ts}")
    
print("DONE!")