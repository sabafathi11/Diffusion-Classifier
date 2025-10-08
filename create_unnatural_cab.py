#!/usr/bin/env python3
"""
Multi-Fruit Image Generator with Unnatural Colors
==================================================

Generates images of fruits in unnatural colors for dataset creation.
Each fruit is generated 35 times with colors that are not its natural color.

Author: AI Assistant
Date: 2024
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import random
import time
from diffusion.utils import DATASET_ROOT

try:
    import openai
    from PIL import Image
    import requests
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages: pip install openai pillow requests")
    sys.exit(1)


# Fruit definitions with their natural colors
FRUIT_COLORS = {
    'cherry': 'red',
    'pomegranate': 'red',
    'strawberry': 'red',
    'tomato': 'red',
    'banana': 'yellow',
    'lemon': 'yellow',
    'corn': 'yellow',
    'broccoli': 'green',
    'cucumber': 'green',
    'brinjal': 'purple',
    'plum': 'purple',
    'orange': 'orange',
    'carrot': 'orange'
}

# Available colors for assignment
AVAILABLE_COLORS = ['yellow', 'red', 'green', 'purple', 'orange']


class ImageGenerator:
    """
    A comprehensive class for generating images using OpenAI's DALL-E API.
    """
    
    SUPPORTED_SIZES = {
        "1024x1024": "1024x1024",
        "1024x1792": "1024x1792", 
        "1792x1024": "1792x1024",
        "512x512": "512x512",
        "256x256": "256x256"
    }
    
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "webp"]
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "generated_images"):
        """Initialize the ImageGenerator."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("ImageGenerator initialized")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate the image generation prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return True
    
    def validate_size(self, size: str) -> str:
        """Validate and normalize image size."""
        if size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported size: {size}. "
                f"Supported sizes: {list(self.SUPPORTED_SIZES.keys())}"
            )
        return self.SUPPORTED_SIZES[size]
    
    def generate_filename(self, fruit: str, color: str, index: int, format: str = "png") -> str:
        """Generate a unique filename for the image."""
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{fruit}_{color}_{index:03d}_{timestamp}.{format}"
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download image from URL and save to file."""
        try:
            print(f"  Downloading: {filename}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            try:
                with Image.open(filepath) as img:
                    img.verify()
                print(f"  ✓ Saved: {filepath}")
                return True
            except Exception as e:
                print(f"  ✗ Invalid image: {e}")
                filepath.unlink()
                return False
                
        except requests.RequestException as e:
            print(f"  ✗ Download failed: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return False
    
    def generate_image(
        self, 
        prompt: str, 
        size: str = "1024x1024", 
        quality: str = "standard",
        style: str = "vivid",
        format: str = "png",
        model: str = "dall-e-3"
    ) -> Dict[str, Any]:
        """Generate a single image using DALL-E API."""
        self.validate_prompt(prompt)
        size = self.validate_size(size)
        
        if format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            image_data = response.data[0]
            
            return {
                "url": image_data.url,
                "revised_prompt": getattr(image_data, 'revised_prompt', None)
            }
            
        except openai.APIError as e:
            print(f"  ✗ OpenAI API Error: {e}")
            raise
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            raise


def get_unnatural_colors(natural_color: str) -> List[str]:
    """Get list of unnatural colors for a fruit."""
    return [c for c in AVAILABLE_COLORS if c != natural_color]


def generate_fruit_dataset(
    generator: ImageGenerator,
    fruits: Dict[str, str],
    images_per_fruit: int = 35,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "natural",
    format: str = "png",
    model: str = "dall-e-3",
    delay: float = 1.0
):
    """
    Generate dataset of fruits with unnatural colors.
    
    Args:
        generator: ImageGenerator instance
        fruits: Dictionary of fruit names and their natural colors
        images_per_fruit: Number of images to generate per fruit
        size: Image size
        quality: Image quality
        style: Image style
        format: Output format
        model: DALL-E model
        delay: Delay between API calls in seconds
    """
    total_fruits = len(fruits)
    total_images = total_fruits * images_per_fruit
    completed = 0
    failed = 0
    
    print(f"\n{'='*70}")
    print(f"FRUIT DATASET GENERATION")
    print(f"{'='*70}")
    print(f"Total fruits: {total_fruits}")
    print(f"Images per fruit: {images_per_fruit}")
    print(f"Total images to generate: {total_images}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for fruit_idx, (fruit, natural_color) in enumerate(fruits.items(), 1):
        print(f"\n[{fruit_idx}/{total_fruits}] Processing: {fruit.upper()} (natural: {natural_color})")
        print(f"{'-'*70}")
        
        unnatural_colors = get_unnatural_colors(natural_color)
        fruit_success = 0
        fruit_failed = 0
        
        for img_idx in range(images_per_fruit):
            # Cycle through unnatural colors
            color = unnatural_colors[img_idx % len(unnatural_colors)]
            
            prompt = (
                f"A highly realistic photograph of a single {color} {fruit}. "
                f"The fruit should look natural and fresh, with realistic texture, "
                f"captured in professional food photography style, with natural lighting."
            )
            
            filename = generator.generate_filename(fruit, color, img_idx + 1, format)
            
            print(f"\n  Image {img_idx + 1}/{images_per_fruit}: {fruit} in {color}")
            
            try:
                # Generate image
                result = generator.generate_image(
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    style=style,
                    format=format,
                    model=model
                )
                
                # Download and save
                success = generator.download_image(result["url"], filename)
                
                if success:
                    completed += 1
                    fruit_success += 1
                else:
                    failed += 1
                    fruit_failed += 1
                
                # Progress update
                progress = (completed + failed) / total_images * 100
                print(f"  Progress: {completed}/{total_images} ({progress:.1f}%) | Failed: {failed}")
                
                # Rate limiting delay
                if img_idx < images_per_fruit - 1:
                    time.sleep(delay)
                    
            except KeyboardInterrupt:
                print("\n\n⚠ Generation interrupted by user!")
                print(f"Completed: {completed}/{total_images}")
                print(f"Failed: {failed}")
                raise
            except Exception as e:
                print(f"  ✗ Error generating image: {e}")
                failed += 1
                fruit_failed += 1
                time.sleep(delay)
        
        print(f"\n{fruit.capitalize()} complete: {fruit_success} success, {fruit_failed} failed")
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total images generated: {completed}/{total_images}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/total_images*100:.1f}%")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Output directory: {generator.output_dir.absolute()}")
    print(f"{'='*70}\n")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate fruit images with unnatural colors for dataset creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python fruit_generator.py --images-per-fruit 35 --model dall-e-3
        """
    )
    
    parser.add_argument("--images-per-fruit", type=int, default=10,
                       help="Number of images per fruit (default: 35)")
    parser.add_argument("--size", default="1024x1024", 
                       choices=["1024x1024", "1024x1792", "1792x1024"],
                       help="Image size (default: 1024x1024)")
    parser.add_argument("--quality", default="standard", choices=["standard", "hd"],
                       help="Image quality (default: standard)")
    parser.add_argument("--style", default="natural", choices=["vivid", "natural"],
                       help="Image style (default: natural)")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "jpeg", "webp"],
                       help="Output format (default: png)")
    parser.add_argument("--output-dir", default="/mnt/public/Ehsan/docker_private/learning2/saba/datasets/fruits_unnatural_colors",
                       help="Output directory (default: generated_images)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="dall-e-3", choices=["dall-e-3", "dall-e-2"],
                       help="DALL-E model to use (default: dall-e-3)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        generator = ImageGenerator(api_key=args.api_key, output_dir=args.output_dir)
        
        generate_fruit_dataset(
            generator=generator,
            fruits=FRUIT_COLORS,
            images_per_fruit=args.images_per_fruit,
            size=args.size,
            quality=args.quality,
            style=args.style,
            format=args.format,
            model=args.model,
            delay=args.delay
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())