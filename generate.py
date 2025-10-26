#!/usr/bin/env python3
"""
Batch Image Generator for Multiple Objects
==========================================

This script generates multiple images for predefined objects organized in folders.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

try:
    import openai
    from PIL import Image
    import requests
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages: pip install openai pillow requests")
    sys.exit(1)


class BatchImageGenerator:
    """
    Generate multiple images for multiple objects organized in folders.
    """
    
    SUPPORTED_SIZES = {
        "1024x1024": "1024x1024",
        "1024x1792": "1024x1792", 
        "1792x1024": "1792x1024",
    }
    
    # Define all objects organized by folders
    OBJECTS = {
        #"folder_1": ["Tree", "Car", "Person", "Fish", "House", "Airplane", "Flower", "Book", "Chair", "Cat" ],
        "folder_2": ["car"],
        #"folder_2": ["Table", "Spoon", "Mug", "Blanket", "Door", "Shoe", "Bag", "Candle", "Ring", "Statue"],
        #"folder_2": [],
        #"folder_3": ["Ball", "Box", "Plate", "Clock", "Window", "Wheel", "Dice", "Coin"],
        #"folder_3": [],
        #"folder_4": ["Elephant", "Ant", "Whale", "Mouse", "Mountain", "Pebble", "Truck", "Key", "Building", "Bird"],
        #"folder_4": [],
        #"folder_5": ["Tea", "Coffee", "Soup", "Ice cube", "Water bottle", "Juice", "Fireplace", "Stove", "Engine", "Ice cream"]
        #"folder_5": [],
    }
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "generated_images"):
        """
        Initialize the BatchImageGenerator.
        
        Args:
            api_key: OpenAI API key
            output_dir: Base directory for all generated images
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("BatchImageGenerator initialized")
        print(f"Base output directory: {self.output_dir.absolute()}")
    
    def create_prompt(self, object_name: str) -> str:
        """
        Create a prompt for an object using the template.
        
        Args:
            object_name: Name of the object
            
        Returns:
            str: Formatted prompt
        """
        return f"A realistic, high-quality image of a {object_name} centered on a plain white background, studio lighting, no shadows, no text, no people, minimalistic composition."
        #return f"A {object_name}, shot on a clean plain background, central composition, highly detailed, realistic lighting, ultra-high resolution, studio quality"
    
    def download_image(self, url: str, filepath: Path) -> bool:
        """
        Download image from URL and save to file.
        
        Args:
            url: Image URL
            filepath: Full path where to save the file
            
        Returns:
            bool: True if successful
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Validate image
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return True
            except Exception as e:
                print(f"Downloaded file is not a valid image: {e}")
                filepath.unlink()
                return False
                
        except Exception as e:
            print(f"Failed to download image: {e}")
            return False
    
    def generate_images_for_object(
        self, 
        object_name: str,
        folder_path: Path,
        num_images: int = 12,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        model: str = "dall-e-3",
        delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate multiple images for a single object.
        
        Args:
            object_name: Name of the object
            folder_path: Path to the object's folder
            num_images: Number of images to generate
            size: Image size
            quality: Image quality
            style: Image style
            model: DALL-E model
            delay: Delay between API calls in seconds
            
        Returns:
            Dict: Generation results and statistics
        """
        prompt = self.create_prompt(object_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Generating {num_images} images for: {object_name}")
        print(f"Folder: {folder_path}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        results = {
            "object": object_name,
            "folder": str(folder_path),
            "total": num_images,
            "successful": 0,
            "failed": 0,
            "images": []
        }
        
        for i in range(num_images):
            try:
                print(f"\nGenerating image {i+1}/{num_images} for {object_name}...")
                
                # Call DALL-E API
                response = self.client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    style=style,
                    n=1
                )
                
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{object_name.lower().replace(' ', '_')}_{i+1:02d}_{timestamp}.png"
                filepath = folder_path / filename
                
                # Download image
                image_url = response.data[0].url
                success = self.download_image(image_url, filepath)
                
                if success:
                    results["successful"] += 1
                    print(f"✓ Saved: {filename}")
                else:
                    results["failed"] += 1
                    print(f"✗ Failed to save: {filename}")
                
                results["images"].append({
                    "index": i + 1,
                    "filename": filename,
                    "success": success,
                    "filepath": str(filepath) if success else None
                })
                
                # Add delay between requests to avoid rate limiting
                if i < num_images - 1:
                    time.sleep(delay)
                
            except openai.APIError as e:
                print(f"✗ API Error for image {i+1}: {e}")
                results["failed"] += 1
                results["images"].append({
                    "index": i + 1,
                    "success": False,
                    "error": str(e)
                })
                time.sleep(delay * 2)  # Longer delay after error
                
            except Exception as e:
                print(f"✗ Unexpected error for image {i+1}: {e}")
                results["failed"] += 1
                results["images"].append({
                    "index": i + 1,
                    "success": False,
                    "error": str(e)
                })
        
        print(f"\nCompleted {object_name}: {results['successful']}/{results['total']} successful")
        return results
    
    def generate_all(
        self,
        num_images_per_object: int = 12,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        model: str = "dall-e-3",
        delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate images for all objects in all folders.
        
        Args:
            num_images_per_object: Number of images per object
            size: Image size
            quality: Image quality
            style: Image style
            model: DALL-E model
            delay: Delay between API calls
            
        Returns:
            Dict: Overall generation statistics
        """
        start_time = datetime.now()
        overall_results = {
            "start_time": start_time.isoformat(),
            "folders": {},
            "total_objects": 0,
            "total_images_requested": 0,
            "total_images_successful": 0,
            "total_images_failed": 0
        }
        
        print(f"\n{'#'*80}")
        print(f"BATCH IMAGE GENERATION STARTED")
        print(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Images per object: {num_images_per_object}")
        print(f"Settings: size={size}, quality={quality}, style={style}, model={model}")
        print(f"{'#'*80}\n")
        
        for folder_name, objects in self.OBJECTS.items():
            print(f"\n{'*'*80}")
            print(f"PROCESSING {folder_name.upper()}")
            print(f"Objects: {len(objects)}")
            print(f"{'*'*80}")
            
            folder_results = {
                "objects": {},
                "total_successful": 0,
                "total_failed": 0
            }
            
            for obj in objects:
                # Create subfolder for each object
                object_folder = self.output_dir / folder_name / obj.lower().replace(' ', '_')
                
                # Generate images
                obj_results = self.generate_images_for_object(
                    object_name=obj,
                    folder_path=object_folder,
                    num_images=num_images_per_object,
                    size=size,
                    quality=quality,
                    style=style,
                    model=model,
                    delay=delay
                )
                
                folder_results["objects"][obj] = obj_results
                folder_results["total_successful"] += obj_results["successful"]
                folder_results["total_failed"] += obj_results["failed"]
                
                overall_results["total_objects"] += 1
                overall_results["total_images_requested"] += obj_results["total"]
                overall_results["total_images_successful"] += obj_results["successful"]
                overall_results["total_images_failed"] += obj_results["failed"]
            
            overall_results["folders"][folder_name] = folder_results
            
            print(f"\n{folder_name} Summary:")
            print(f"  Successful: {folder_results['total_successful']}")
            print(f"  Failed: {folder_results['total_failed']}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        overall_results["end_time"] = end_time.isoformat()
        overall_results["duration_seconds"] = duration.total_seconds()
        
        print(f"\n{'#'*80}")
        print(f"BATCH GENERATION COMPLETE")
        print(f"Duration: {duration}")
        print(f"Total Objects: {overall_results['total_objects']}")
        print(f"Total Images Requested: {overall_results['total_images_requested']}")
        print(f"Total Successful: {overall_results['total_images_successful']}")
        print(f"Total Failed: {overall_results['total_images_failed']}")
        print(f"Success Rate: {overall_results['total_images_successful']/overall_results['total_images_requested']*100:.1f}%")
        print(f"{'#'*80}\n")
        
        return overall_results


def main():
    """
    Main function to run batch generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate multiple images for multiple objects organized in folders"
    )
    parser.add_argument("--num-images", type=int, default=1,
                       help="Number of images per object")
    parser.add_argument("--size", default="1024x1024",
                       choices=["1024x1024", "1024x1792", "1792x1024"],
                       help="Image size (default: 1024x1024)")
    parser.add_argument("--quality", default="standard", choices=["standard", "hd"],
                       help="Image quality (default: standard)")
    parser.add_argument("--style", default="natural", choices=["vivid", "natural"],
                       help="Image style (default: natural)")
    parser.add_argument("--output-dir", default="/mnt/public/Ehsan/docker_private/learning2/saba/datasets/attrs2",
                       help="Base output directory (default: generated_images)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="dall-e-3", choices=["dall-e-3", "dall-e-2"],
                       help="DALL-E model (default: dall-e-3)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        generator = BatchImageGenerator(
            api_key=args.api_key,
            output_dir=args.output_dir
        )
        
        results = generator.generate_all(
            num_images_per_object=args.num_images,
            size=args.size,
            quality=args.quality,
            style=args.style,
            model=args.model,
            delay=args.delay
        )
        
        # Save results to JSON file
        import json
        results_file = Path(args.output_dir) / f"generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nGeneration cancelled by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())