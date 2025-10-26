# # import os
# # import shutil
# # from pathlib import Path
# # from collections import defaultdict

# # def get_image_type(filename):
# #     """Extract the object type from filename (e.g., 'blender' from 'blender_snowboard_clock.png')"""
# #     # Split by underscore and take first part to identify the type
# #     parts = filename.split('_')
# #     return parts[0]

# # def copy_first_n_images_by_type(source_folders, destination_folder, n=10):
# #     """
# #     Copy the first n images of each type from source folders to destination folder.
    
# #     Args:
# #         source_folders: List of source folder paths
# #         destination_folder: Destination folder path
# #         n: Number of images per type to copy (default: 10)
# #     """
# #     # Create destination folder if it doesn't exist
# #     os.makedirs(destination_folder, exist_ok=True)
    
# #     # Supported image extensions
# #     image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    
# #     # Process each source folder
# #     for folder_idx, source_folder in enumerate(source_folders, 1):
# #         if not os.path.exists(source_folder):
# #             print(f"Warning: Source folder '{source_folder}' does not exist. Skipping...")
# #             continue
        
# #         print(f"\nProcessing folder {folder_idx}: {source_folder}")
        
# #         # Group images by type
# #         images_by_type = defaultdict(list)
        
# #         # Get all image files from the source folder
# #         for filename in sorted(os.listdir(source_folder)):
# #             file_path = os.path.join(source_folder, filename)
            
# #             # Check if it's a file and has an image extension
# #             if os.path.isfile(file_path):
# #                 ext = os.path.splitext(filename)[1].lower()
# #                 if ext in image_extensions:
# #                     image_type = get_image_type(filename)
# #                     images_by_type[image_type].append(filename)
        
# #         # Copy first n images of each type
# #         total_copied = 0
# #         for image_type, files in sorted(images_by_type.items()):
# #             # Take only first n files of this type
# #             files_to_copy = files[:n]
            
# #             print(f"  Type '{image_type}': Found {len(files)} images, copying {len(files_to_copy)}")
            
# #             for filename in files_to_copy:
# #                 src_path = os.path.join(source_folder, filename)
                
# #                 # Create unique filename with folder prefix to avoid conflicts
# #                 dst_filename = f"folder{folder_idx}_{filename}"
# #                 dst_path = os.path.join(destination_folder, dst_filename)
                
# #                 shutil.copy2(src_path, dst_path)
# #                 total_copied += 1
        
# #         print(f"  Total copied from this folder: {total_copied} images")


# # if __name__ == "__main__":
# #     # Define your source folders
# #     source_folders = [
# #         "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/comco/CoCo-FiveObject-MBig",
# #         #"/mnt/public/Ehsan/docker_private/learning2/saba/datasets/comco/CoCo-FiveObject-UR-Big",
# #     ]
    
# #     # Define destination folder
# #     destination_folder = "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/comco/5/middle"
    
# #     # Copy first 10 images of each type from each folder
# #     copy_first_n_images_by_type(source_folders, destination_folder, n=10)
    
# #     print(f"\n✓ All images copied to '{destination_folder}'")

# #!/usr/bin/env python3
# """
# Image Generator using OpenAI's DALL-E API
# ==========================================

# This script demonstrates how to generate and save images using OpenAI's DALL-E API.
# It includes comprehensive error handling, image validation, and flexible saving options.

# Features:
# - Generate images from text prompts
# - Save images in multiple formats (PNG, JPEG, WebP)
# - Automatic filename generation with timestamps
# - Comprehensive error handling
# - Image size and quality validation
# - Batch image generation support

# Requirements:
# - OpenAI API key
# - Python 3.7+
# - Required packages: openai, pillow, requests

# Author: AI Assistant
# Date: 2024
# """

# import os
# import sys
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, List, Dict, Any
# import argparse

# try:
#     import openai
#     from PIL import Image
#     import requests
# except ImportError as e:
#     print(f"Missing required package: {e}")
#     print("Please install required packages: pip install openai pillow requests")
#     sys.exit(1)


# class ImageGenerator:
#     """
#     A comprehensive class for generating images using OpenAI's DALL-E API.
    
#     This class handles all aspects of image generation including:
#     - API communication
#     - Image downloading and saving
#     - Error handling and validation
#     - File management
#     """
    
#     # Supported image sizes for DALL-E
#     SUPPORTED_SIZES = {
#         "1024x1024": "1024x1024",
#         "1024x1792": "1024x1792", 
#         "1792x1024": "1792x1024",
#         "512x512": "512x512",  # Legacy size
#         "256x256": "256x256"   # Legacy size
#     }
    
#     # Supported image formats
#     SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "webp"]
    
#     def __init__(self, api_key: Optional[str] = None, output_dir: str = "generated_images"):
#         """
#         Initialize the ImageGenerator.
        
#         Args:
#             api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
#             output_dir: Directory to save generated images
#         """
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         if not self.api_key:
#             raise ValueError(
#                 "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
#                 "or pass api_key parameter."
#             )
        
#         # Initialize OpenAI client
#         self.client = openai.OpenAI(api_key=self.api_key)
        
#         # Set up output directory
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
        
#         print("ImageGenerator initialized")
#         print(f"Output directory: {self.output_dir.absolute()}")
    
#     def validate_prompt(self, prompt: str) -> bool:
#         """
#         Validate the image generation prompt.
        
#         Args:
#             prompt: Text prompt for image generation
            
#         Returns:
#             bool: True if prompt is valid
#         """
#         if not prompt or not prompt.strip():
#             raise ValueError("Prompt cannot be empty")
        
#         return True
    
#     def validate_size(self, size: str) -> str:
#         """
#         Validate and normalize image size.
        
#         Args:
#             size: Image size string
            
#         Returns:
#             str: Validated size string
#         """
#         if size not in self.SUPPORTED_SIZES:
#             raise ValueError(
#                 f"Unsupported size: {size}. "
#                 f"Supported sizes: {list(self.SUPPORTED_SIZES.keys())}"
#             )
#         return self.SUPPORTED_SIZES[size]
    
#     def generate_filename(self, prompt: str, format: str = "png", index: int = 0) -> str:
#         """
#         Generate a unique filename for the image.
        
#         Args:
#             prompt: Original prompt
#             format: Image format
#             index: Index for batch generation
            
#         Returns:
#             str: Generated filename
#         """
#         # Create a safe filename from prompt
#         safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
#         safe_prompt = safe_prompt.replace(' ', '_')
        
#         # Add timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Add index if batch generation
#         index_suffix = f"_{index}" if index > 0 else ""
        
#         return f"{safe_prompt}_{timestamp}{index_suffix}.{format}"
    
#     def download_image(self, url: str, filename: str) -> bool:
#         """
#         Download image from URL and save to file.
        
#         Args:
#             url: Image URL
#             filename: Local filename to save
            
#         Returns:
#             bool: True if successful
#         """
#         try:
#             print(f"Downloading image from: {url}")
            
#             # Download image
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
            
#             # Save image
#             filepath = self.output_dir / filename
#             with open(filepath, 'wb') as f:
#                 f.write(response.content)
            
#             # Validate image
#             try:
#                 with Image.open(filepath) as img:
#                     img.verify()
#                 print(f"Image saved: {filepath}")
#                 return True
#             except Exception as e:
#                 print(f"Downloaded file is not a valid image: {e}")
#                 filepath.unlink()  # Remove invalid file
#                 return False
                
#         except requests.RequestException as e:
#             print(f"Failed to download image: {e}")
#             return False
#         except Exception as e:
#             print(f"Unexpected error downloading image: {e}")
#             return False
    
#     def generate_image(
#         self, 
#         prompt: str, 
#         size: str = "1024x1024", 
#         quality: str = "standard",
#         style: str = "vivid",
#         n: int = 1,
#         format: str = "png",
#         model: str = "dall-e-3"
#     ) -> List[Dict[str, Any]]:
#         """
#         Generate image(s) using DALL-E API.
        
#         Args:
#             prompt: Text description of the image
#             size: Image size (1024x1024, 1024x1792, 1792x1024)
#             quality: Image quality (standard, hd)
#             style: Image style (vivid, natural)
#             n: Number of images to generate (1-10)
#             format: Output format (png, jpg, webp)
#             model: DALL-E model to use (dall-e-3, dall-e-2)
            
#         Returns:
#             List[Dict]: List of generation results with metadata
#         """
#         # Validate inputs
#         self.validate_prompt(prompt)
#         size = self.validate_size(size)
        
#         if n < 1 or n > 10:
#             raise ValueError("Number of images must be between 1 and 10")
        
#         if format.lower() not in self.SUPPORTED_FORMATS:
#             raise ValueError(f"Unsupported format: {format}")
        
#         print(f"Generating {n} image(s) with prompt: '{prompt}'")
#         print(f"Size: {size}, Quality: {quality}, Style: {style}, Model: {model}")
        
#         try:
#             # Call DALL-E API
#             response = self.client.images.generate(
#                 model=model,
#                 prompt=prompt,
#                 size=size,
#                 quality=quality,
#                 style=style,
#                 n=n
#             )
            
#             results = []
#             for i, image_data in enumerate(response.data):
#                 # Generate filename
#                 filename = self.generate_filename(prompt, format, i)
                
#                 # Download and save image
#                 success = self.download_image(image_data.url, filename)
                
#                 result = {
#                     "prompt": prompt,
#                     "filename": filename,
#                     "url": image_data.url,
#                     "size": size,
#                     "quality": quality,
#                     "style": style,
#                     "model": model,
#                     "success": success,
#                     "filepath": str(self.output_dir / filename) if success else None
#                 }
#                 results.append(result)
                
#                 if success:
#                     print(f"Image {i+1}/{n} generated")
#                 else:
#                     print(f"Failed to save image {i+1}/{n}")
            
#             return results
            
#         except openai.APIError as e:
#             print(f"OpenAI API Error: {e}")
#             raise
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             raise


# def main():
#     """
#     Main function with command-line interface.
#     """
#     parser = argparse.ArgumentParser(
#         description="Generate images using OpenAI's DALL-E API",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python image_generator.py "A beautiful sunset over mountains"
#   python image_generator.py "A cute cat" --size 1024x1792 --quality hd --model dall-e-3
#   python image_generator.py "Abstract art" --n 3 --model dall-e-2
#         """
#     )
    
#     parser.add_argument("prompt", nargs="?",
#                         # , shot on a clean plain background, central composition, highly detailed, realistic lighting, ultra-high resolution, studio quality
#                          default="A car",
#                          help="Text prompt for image generation")
#     parser.add_argument("--size", default="1024x1024", 
#                        choices=["1024x1024", "1024x1792", "1792x1024"],
#                        help="Image size (default: 1024x1024)")
#     parser.add_argument("--quality", default="standard", choices=["standard", "hd"],
#                        help="Image quality (default: standard)")
#     parser.add_argument("--style", default="natural", choices=["vivid", "natural"],
#                        help="Image style (default: natural)")
#     parser.add_argument("--n", type=int, default=1, help="Number of images (1-10, default: 1)")
#     parser.add_argument("--format", default="png", choices=["png", "jpg", "jpeg", "webp"],
#                        help="Output format (default: png)")
#     parser.add_argument("--output-dir", default="generated_images",
#                        help="Output directory (default: generated_images)")
#     parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
#     parser.add_argument("--model", default="dall-e-3", choices=["dall-e-3", "dall-e-2"],
#                        help="DALL-E model to use (default: dall-e-3)")
    
#     args = parser.parse_args()
    
#     try:
#         # Initialize generator
#         generator = ImageGenerator(api_key=args.api_key, output_dir=args.output_dir)
        
#         if args.prompt:
#             # Single prompt generation
#             results = generator.generate_image(
#                 prompt=args.prompt,
#                 size=args.size,
#                 quality=args.quality,
#                 style=args.style,
#                 n=args.n,
#                 format=args.format,
#                 model=args.model
#             )
#         else:
#             print("Please provide a prompt")
#             parser.print_help()
#             return 1
        
#         # Print outputs
#         if results:
#             successful = sum(1 for r in results if r.get("success", False))
#             print(f"Generated {successful}/{len(results)} images")
#             for r in results:
#                 if r.get("success"):
#                     print(f"Saved: {r.get('filepath')}")
        
#         return 0
        
#     except KeyboardInterrupt:
#         print("Generation cancelled by user")
#         return 1
#     except Exception as e:
#         print(f"Error: {e}")
#         return 1


# if __name__ == "__main__":
#     sys.exit(main())


from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
import numpy as np

# Load your existing image
image_path = "folder1_apple_bed.png"
image_pil = Image.open(image_path).convert("RGB").resize((1024, 1024))

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
).to(device)


prompt = "a red apple and a bed"
gen = set_seed(0)

label = 'red'

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(
            prompt=prompt,
            image=image_pil,
            strength=0.03,      # tiny value to avoid changes but keep pipeline working
            num_inference_steps=150,
            generator=gen
        )

        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(label)

        heat_map.plot_overlay(out.images[0])
        plt.savefig(f'heat_map_{label}2.png')

