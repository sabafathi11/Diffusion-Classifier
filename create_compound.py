import os
import random
from pathlib import Path
from PIL import Image
import argparse

# Define attributes for each object in each category
ATTRIBUTES = {
    "Part-Whole": {
        "Tree": "leafy",
        "Car": "wheeled",
        "bird": "feathered",
        "fish": "finned",
        "House": "windowed",
        "Airplane": "winged",
        "Flower": "petaled",
        "book": "paged",
        "Chair": "legged",
        "Cat": "tailed"
    },
    "Shape": {
        "Ball": "circular",
        "Plate": "circular",
        "Clock": "circular",
        "Wheel": "circular",
        "Coin": "circular",
        "Box": "rectangular",
        "Window": "rectangular",
        "Book": "rectangular",
        "Table": "rectangular",
        "Door": "rectangular"
    },
    "Material & Texture": {
        "Table": "wooden",
        "Spoon": "silver",
        "Mug": "ceramic",
        "Blanket": "woolen",
        "Door": "wooden",
        "Shoe": "leather",
        "Bag": "fabric",
        "Candle": "wax",
        "Ring": "golden",
        "Statue": "marble"
    },
    "Size": {
        "Elephant": "big",
        "Whale": "big",
        "Truck": "big",
        "Building": "big",
        "wind turbine": "big",
        "Ant": "small",
        "Mouse": "small",
        "Pebble": "small",
        "Key": "small",
        "Bird": "small"
    },
    "Temperature": {
        "Tea": "hot",
        "Coffee": "hot",
        "Soup": "hot",
        "Ice cube": "cold",
        "Water bottle": "cold",
        "Juice": "cold",
        "Fireplace": "hot",
        "stove": "hot",
        "engine": "hot",
        "Ice cream": "cold"
    }
}

# Define similar attributes that shouldn't be paired
SIMILAR_ATTRIBUTES = {
    "Part-Whole": [
        {"leafy","petaled"},
        {"feathered", "finned"},
    ],
    "Shape": [
        {"circular"},
        {"rectangular"}
    ],
    "Size": [
        {"big"},
        {"small"}
    ],
    "Temperature": [
        {"hot"},
        {"cold"}
    ]
}

def are_attributes_similar(attr1, attr2, category):
    """Check if two attributes are too similar to be paired."""
    if attr1 == attr2:
        return True
    
    if category in SIMILAR_ATTRIBUTES:
        for group in SIMILAR_ATTRIBUTES[category]:
            if attr1 in group and attr2 in group:
                return True
    
    return False

def get_all_images(obj_folder):
    """Get all images from an object folder."""
    images = []
    for file in sorted(os.listdir(obj_folder)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(obj_folder, file))
    return images

def create_compound_image(img1_path, img2_path, output_path):
    """Create a compound image by placing two images side by side."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Calculate dimensions for the compound image
    max_height = max(img1.height, img2.height)
    total_width = img1.width + img2.width
    
    # Create new image with white background
    compound = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images side by side
    compound.paste(img1, (0, 0))
    compound.paste(img2, (img1.width, 0))
    
    # Save compound image
    compound.save(output_path)
    
    return img1_path, img2_path

def process_category(category_name, input_base, output_base):
    """Process a single category to create compound images."""
    category_path = os.path.join(input_base, category_name)
    
    if not os.path.exists(category_path):
        print(f"Warning: Category '{category_name}' not found in input directory")
        return
    
    # Create output directories
    output_category = os.path.join(output_base, category_name)
    single_dir = os.path.join(output_category, "single")
    compound_dir = os.path.join(output_category, "compound")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(compound_dir, exist_ok=True)
    
    # Get all object folders and their images
    objects = {}
    for obj_name in os.listdir(category_path):
        obj_path = os.path.join(category_path, obj_name)
        if os.path.isdir(obj_path):
            # Find matching attribute
            matching_key = None
            for attr_key in ATTRIBUTES[category_name]:
                if attr_key.lower() == obj_name.lower():
                    matching_key = attr_key
                    break
            
            if matching_key:
                img_paths = get_all_images(obj_path)
                if img_paths:
                    objects[matching_key] = {
                        'paths': img_paths,
                        'attribute': ATTRIBUTES[category_name][matching_key],
                        'name': obj_name
                    }
    
    if len(objects) < 2:
        print(f"Warning: Not enough objects found in '{category_name}'")
        return
    
    # First, copy ALL single images to single directory
    print(f"\n{category_name}: Copying all single images...")
    single_counters = {}
    all_single_images = {}
    
    for obj_key, obj in objects.items():
        attr_obj_key = f"{obj['attribute']}_{obj['name']}"
        single_counters[attr_obj_key] = 0
        
        for img_path in obj['paths']:
            single_counters[attr_obj_key] += 1
            single_filename = f"{obj['attribute']}_{obj['name']}_{single_counters[attr_obj_key]:03d}{Path(img_path).suffix}"
            single_output = os.path.join(single_dir, single_filename)
            Image.open(img_path).save(single_output)
            all_single_images[img_path] = single_counters[attr_obj_key]
    
    total_singles = sum(len(obj['paths']) for obj in objects.values())
    print(f"  Copied {total_singles} single images")
    
    # Generate all possible image pair combinations
    obj_keys = list(objects.keys())
    all_combinations = []
    
    for i, obj1_key in enumerate(obj_keys):
        for obj2_key in obj_keys[i+1:]:
            attr1 = objects[obj1_key]['attribute']
            attr2 = objects[obj2_key]['attribute']
            
            # Check if attributes are different enough
            if not are_attributes_similar(attr1, attr2, category_name):
                # Create all combinations of images between these two objects
                for img1_path in objects[obj1_key]['paths']:
                    for img2_path in objects[obj2_key]['paths']:
                        all_combinations.append({
                            'obj1_key': obj1_key,
                            'obj2_key': obj2_key,
                            'img1_path': img1_path,
                            'img2_path': img2_path
                        })
    
    # Use ALL combinations (no limit)
    print(f"  Creating {len(all_combinations)} compound images")
    
    # Create compound images
    compound_counter = 1
    for idx, combo in enumerate(all_combinations):
        obj1_key = combo['obj1_key']
        obj2_key = combo['obj2_key']
        obj1 = objects[obj1_key]
        obj2 = objects[obj2_key]
        img1_path = combo['img1_path']
        img2_path = combo['img2_path']
        
        # Create compound image
        compound_filename = f"{obj1['attribute']}_{obj1['name']}_{obj2['attribute']}_{obj2['name']}_{compound_counter:03d}.png"
        compound_output = os.path.join(compound_dir, compound_filename)
        
        create_compound_image(img1_path, img2_path, compound_output)
        compound_counter += 1
        
        if (idx + 1) % 100 == 0 or idx == len(all_combinations) - 1:
            print(f"  Created {idx + 1}/{len(all_combinations)} compounds...")

def main():
    parser = argparse.ArgumentParser(description='Generate compound images from object pairs')
    parser.add_argument('--input', default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attrs-nobg', 
                        help='Input directory containing category folders')
    parser.add_argument('--output', default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attributes', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    input_base = args.input
    output_base = args.output
    
    # Process each category
    categories = ["Part-Whole", "Shape", "Material & Texture", "Size", "Temperature"]
    
    print(f"Starting compound image generation...")
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    print(f"Mode: Using ALL images and creating ALL valid compound combinations")
    
    for category in categories:
        process_category(category, input_base, output_base)
    
    print("\n✓ All compound images created successfully!")

if __name__ == "__main__":
    main()