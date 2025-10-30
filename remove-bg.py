import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
import torchvision.transforms as torch_transforms
from tqdm import tqdm


class BackgroundRemover:
    """Wrapper for BiRefNet background removal."""
    
    def __init__(self, device="cuda"):
        print("Loading BiRefNet for background removal...")
        torch.set_float32_matmul_precision("high")
        
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(device)
        
        self.transform_image = torch_transforms.Compose([
            torch_transforms.Resize((1024, 1024)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.device = device
        print("BiRefNet loaded successfully")
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from a PIL image and replace with white."""
        # Convert to RGB FIRST (handles RGBA, LA, L, etc.)
        if image.mode != 'RGB':
            # If image has transparency, composite on white background first
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            else:
                image = image.convert('RGB')
        
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = torch_transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        
        # Create white background
        white_background = Image.new('RGB', image_size, (255, 255, 255))
        
        # Composite the image onto white background using the mask
        white_background.paste(image, mask=mask)
        
        return white_background


def process_directory(input_dir, output_dir, bg_remover):
    """
    Process all PNG images in input_dir and save to output_dir with same structure.
    
    Args:
        input_dir: Source directory (e.g., 'other-attributes')
        output_dir: Destination directory (e.g., 'other-attributes-nobg')
        bg_remover: BackgroundRemover instance
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all PNG images
    png_files = list(input_path.rglob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG images to process")
    
    # Track statistics
    success_count = 0
    error_count = 0
    
    # Process each image
    for img_path in tqdm(png_files, desc="Processing images"):
        # Calculate relative path from input_dir
        relative_path = img_path.relative_to(input_path)
        
        # Create corresponding output path
        output_img_path = output_path / relative_path
        
        # Create output directory if it doesn't exist
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load image
            image = Image.open(img_path)
            
            # Remove background
            result = bg_remover.remove_background(image)
            
            # Save result
            result.save(output_img_path)
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(png_files)}")
    print(f"Errors: {error_count}/{len(png_files)}")
    print(f"Results saved to {output_dir}")


def main():
    # Configuration
    input_directory = "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attrs"
    output_directory = "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attrs-nobg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Initialize background remover
    bg_remover = BackgroundRemover(device=device)
    
    # Process all images
    process_directory(input_directory, output_directory, bg_remover)


if __name__ == "__main__":
    main()