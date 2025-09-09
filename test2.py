from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
import numpy as np
from diffusion.datasets import get_target_dataset
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_pil_image
import os.path as osp
import os
import itertools


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

def run_experiment(pipe, image_pil, prompt, classes, hyperparams, output_dir, img_idx, label):
    num_steps, strength, threshold_percentile = hyperparams
    
    param_dir = f"steps_{num_steps}_strength_{strength:.3f}_thresh_{threshold_percentile}"
    full_output_dir = osp.join(output_dir, label, param_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    gen = set_seed(0)
    
    # Store heatmaps for all classes
    heatmaps = []

    with torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(
                prompt=prompt,
                image=image_pil,
                strength=strength,
                num_inference_steps=num_steps,
                generator=gen
            )
            
            for class_name in classes:
                heat_map = tc.compute_global_heat_map().compute_word_heat_map(class_name)
                hm = np.array(heat_map.heatmap.cpu())
                threshold = np.percentile(hm, threshold_percentile)
                hm[hm < threshold] = 0  # filter weak activations
                heatmaps.append(hm)
    
    heatmaps = np.stack(heatmaps, axis=0)  # shape: (num_classes, H, W)
    class_map = np.argmax(heatmaps, axis=0)  # shape: (H, W)
    
    # Color map for visualization
    colors = plt.cm.get_cmap('tab10', len(classes))
    segmentation = colors(class_map / len(classes))  # RGBA
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].imshow(image_pil)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(class_map, cmap='tab10')
    axes[1].set_title('Class Map')
    axes[1].axis('off')
    
    axes[2].imshow(image_pil)
    axes[2].imshow(segmentation, alpha=0.5)
    axes[2].set_title('Segmentation Overlay')
    axes[2].axis('off')

    fig.suptitle(f'Steps: {num_steps}, Strength: {strength}, Threshold: {threshold_percentile}%, Image: {img_idx}')
    
    filename = f'segmentation_img{img_idx}_steps{num_steps}_str{strength:.3f}_thr{threshold_percentile}.png'
    plt.savefig(osp.join(full_output_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Setup
    interpolation = INTERPOLATIONS['bicubic']
    transform = get_transform(interpolation, 512)
    dataset = get_target_dataset('cifar10', 'test', transform=transform)
    
    rand_idx = np.random.choice(np.arange(0, len(dataset)), size=5, replace=False)
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    device = 'cuda'
    
    # Load model once
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
    ).to(device)

    class_names = dataset.classes
    prompt = ' '.join(class_names)

    # Hyperparameter configurations
    HYPERPARAMS = {
        'num_inference_steps': [20, 50, 100],
        'strength': [0.01, 0.03, 0.05, 0.1],
        'threshold_percentile': [70, 80, 90, 95]
    }
    
    # Create all combinations of hyperparameters
    param_combinations = list(itertools.product(
        HYPERPARAMS['num_inference_steps'],
        HYPERPARAMS['strength'], 
        HYPERPARAMS['threshold_percentile']
    ))
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations on {len(rand_idx)} images")
    
    # Main experiment loop
    for i in rand_idx:
        img, label = dataset[i]
        label = class_names[label]
        print(f"\nProcessing Image Index: {i}, Label: {label}, Image shape: {img.shape}")
        image_pil = to_pil_image(img)
        
        for param_idx, params in enumerate(param_combinations):
            print(f"  Running combination {param_idx + 1}/{len(param_combinations)}: "
                  f"steps={params[0]}, strength={params[1]}, threshold={params[2]}%")
            
            run_experiment(
                pipe, image_pil, prompt, class_names, params, 
                'daam_hyperparameter', i, label
            )
    
    print("\nHyperparameter sweep completed!")
    print(f"Results saved in 'daam_hyperparameter' directory")

if __name__ == "__main__":
    main()


# Alternative: Run specific hyperparameter combinations
def run_custom_params():
    """
    Alternative function to test specific hyperparameter combinations
    Modify the custom_params list below to test specific combinations
    """
    # Define custom parameter combinations
    custom_params = [
        (20, 0.01, 80),   # (num_steps, strength, threshold_percentile)
        (50, 0.03, 80),
        (100, 0.05, 90),
        # Add more combinations as needed
    ]
    
    # Setup (same as main)
    interpolation = INTERPOLATIONS['bicubic']
    transform = get_transform(interpolation, 512)
    dataset = get_target_dataset('cifar10', 'test', transform=transform)
    
    rand_idx = np.random.choice(np.arange(0, len(dataset)), size=2, replace=False)  # Fewer images for testing
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    device = 'cuda'
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
    ).to(device)
    
    prompt = "bird horse cat deer frog dog truck airplane automobile ship"
    classes = prompt.split()
    
    for i in rand_idx:
        img, label = dataset[i]
        print(f"\nProcessing Image Index: {i}, Label: {label}")
        image_pil = to_pil_image(img)
        
        for params in custom_params:
            print(f"  Testing: steps={params[0]}, strength={params[1]}, threshold={params[2]}%")
            run_experiment(
                pipe, image_pil, prompt, classes, params,
                'daam_custom_params', i, label
            )

# Uncomment the line below to run custom parameters instead of full sweep
# run_custom_params()