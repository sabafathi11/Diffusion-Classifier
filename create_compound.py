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
        "Bird": "feathered", 
        "Fish": "finned", 
        "House": "windowed",
        "Airplane": "winged",
        "Flower": "petaled",
        "Book": "paged", 
        "Chair": "legged",
        "Cat": "tailed"
    },
    "Shape": {
        "Ball": "round",
        "Plate": "round",
        "Clock": "round",
        "Wheel": "round",
        "Coin": "round",
        "Box": "square",
        "Window": "square",
        "Book": "square",
        "Table": "square",
        "Dice": "square"
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
        "Wind turbine": "big",
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
        "Stove": "hot", 
        "Engine": "hot",
        "Ice cream": "cold"
    }
}

# Define similar attributes that shouldn't be paired
# Define attributes for each object in each category
ATTRIBUTES = {
    "Part-Whole": {
        "Tree": "leafy",
        "Car": "wheeled",
        "Bird": "feathered", 
        "Fish": "finned", 
        "House": "windowed",
        "Airplane": "winged",
        "Flower": "petaled",
        "Book": "paged", 
        "Chair": "legged",
        "Cat": "tailed"
    },
    "Shape": {
        "Ball": "round",
        "Plate": "round",
        "Clock": "round",
        "Wheel": "round",
        "Coin": "round",
        "Box": "square",
        "Window": "square",
        "Book": "square",
        "Table": "square",
        "Dice": "square"
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
        "Wind turbine": "big",
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
        "Stove": "hot", 
        "Engine": "hot",
        "Ice cream": "cold"
    }
}

# Define similar attributes that shouldn't be paired
SIMILAR_ATTRIBUTES = {
    "Part-Whole": [
        {"leafy", "petaled"},  # Both are plant features
        {"feathered", "finned"},  # Both are animal coverings
        {"feathered", "winged"},  # Birds have both features
        {"tailed", "finned"},  # Fish have both features
        {"tailed", "legged"},  # Cats and Birds have both features
        {"feathered", "legged"},  # Birds have both features
    ],
    "Shape": [
        {"round"},
        {"square"}
    ],
    "Material & Texture": [
        {"fabric", "leather"},  # Bags can be either fabric or leather
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

def normalize_name(name):
    """Normalize a name for comparison (lowercase, replace underscores with spaces)."""
    return name.lower().replace('_', ' ').strip()

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

def sanitize_filename(name):
    """Convert name to safe filename format (replace spaces with underscores)."""
    return name.replace(' ', '_')

def process_category(category_name, input_base, output_base, max_compounds=500):
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
    skipped_folders = []
    
    for obj_name in os.listdir(category_path):
        obj_path = os.path.join(category_path, obj_name)
        if os.path.isdir(obj_path):
            # Find matching attribute using normalized names
            matching_key = None
            obj_name_normalized = normalize_name(obj_name)
            
            for attr_key in ATTRIBUTES[category_name]:
                attr_key_normalized = normalize_name(attr_key)
                if attr_key_normalized == obj_name_normalized:
                    matching_key = attr_key
                    break
            
            if matching_key:
                img_paths = get_all_images(obj_path)
                if img_paths:
                    objects[matching_key] = {
                        'paths': img_paths,
                        'attribute': ATTRIBUTES[category_name][matching_key],
                        'name': matching_key,  # Use the standardized name from ATTRIBUTES
                        'folder_name': obj_name  # Keep original folder name for reference
                    }
                else:
                    skipped_folders.append(f"{obj_name} (no images found)")
            else:
                # Count images in skipped folder
                img_count = len(get_all_images(obj_path))
                skipped_folders.append(f"{obj_name} ({img_count} images, not in ATTRIBUTES)")
    
    # Report skipped folders
    if skipped_folders:
        print(f"\n{category_name}: WARNING - Skipped folders:")
        for folder_info in skipped_folders:
            print(f"  - {folder_info}")
    
    if len(objects) < 2:
        print(f"Warning: Not enough objects found in '{category_name}'")
        return
    
    # First, copy ALL single images to single directory
    print(f"\n{category_name}: Copying all single images...")
    single_counters = {}
    
    for obj_key, obj in objects.items():
        # Use sanitized filenames
        attr_safe = sanitize_filename(obj['attribute'])
        name_safe = sanitize_filename(obj['name'])
        attr_obj_key = f"{attr_safe}_{name_safe}"
        single_counters[attr_obj_key] = 0
        
        for img_path in obj['paths']:
            single_counters[attr_obj_key] += 1
            single_filename = f"{attr_safe}_{name_safe}_{single_counters[attr_obj_key]:03d}{Path(img_path).suffix}"
            single_output = os.path.join(single_dir, single_filename)
            Image.open(img_path).save(single_output)
    
    total_singles = sum(len(obj['paths']) for obj in objects.values())
    print(f"  Copied {total_singles} single images")
    
    # Generate valid object pairs
    obj_keys = list(objects.keys())
    valid_pairs = []
    
    for i, obj1_key in enumerate(obj_keys):
        for obj2_key in obj_keys[i+1:]:
            attr1 = objects[obj1_key]['attribute']
            attr2 = objects[obj2_key]['attribute']
            
            # Check if attributes are different enough
            if not are_attributes_similar(attr1, attr2, category_name):
                valid_pairs.append((obj1_key, obj2_key))
    
    if not valid_pairs:
        print(f"  No valid object pairs found for {category_name}")
        return
    
    print(f"  Found {len(valid_pairs)} valid object pairs")
    print(f"  Creating {max_compounds} balanced compound images...")
    
    # Calculate how many compounds per pair to balance distribution
    compounds_per_pair = max_compounds // len(valid_pairs)
    remaining = max_compounds % len(valid_pairs)
    
    # Create compound images with balanced distribution
    compound_counter = 1
    created_count = 0
    
    for pair_idx, (obj1_key, obj2_key) in enumerate(valid_pairs):
        obj1 = objects[obj1_key]
        obj2 = objects[obj2_key]
        
        # Determine how many compounds for this pair
        num_for_this_pair = compounds_per_pair + (1 if pair_idx < remaining else 0)
        
        # Get all possible image combinations for this pair
        all_combos = []
        for img1_path in obj1['paths']:
            for img2_path in obj2['paths']:
                all_combos.append((img1_path, img2_path))
        
        # Sample or use all combinations
        if len(all_combos) <= num_for_this_pair:
            selected_combos = all_combos
        else:
            selected_combos = random.sample(all_combos, num_for_this_pair)
        
        # Create compound images for this pair
        for img1_path, img2_path in selected_combos:
            # Use sanitized filenames
            attr1_safe = sanitize_filename(obj1['attribute'])
            name1_safe = sanitize_filename(obj1['name'])
            attr2_safe = sanitize_filename(obj2['attribute'])
            name2_safe = sanitize_filename(obj2['name'])
            
            compound_filename = f"{attr1_safe}_{name1_safe}_{attr2_safe}_{name2_safe}_{compound_counter:03d}.png"
            compound_output = os.path.join(compound_dir, compound_filename)
            
            create_compound_image(img1_path, img2_path, compound_output)
            compound_counter += 1
            created_count += 1
            
            if created_count % 100 == 0:
                print(f"  Created {created_count}/{max_compounds} compounds...")
    
    print(f"  ✓ Created {created_count} compound images with balanced object distribution")

def main():
    parser = argparse.ArgumentParser(description='Generate compound images from object pairs')
    parser.add_argument('--input', default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attrs-nobg', 
                        help='Input directory containing category folders')
    parser.add_argument('--output', default='/mnt/public/Ehsan/docker_private/learning2/saba/datasets/other-attributes', 
                        help='Output directory for results')
    parser.add_argument('--max-compounds', type=int, default=500,
                        help='Maximum number of compound images per category (default: 500)')
    
    args = parser.parse_args()
    
    input_base = args.input
    output_base = args.output
    
    # Process each category
    categories = ["Shape", "Temperature", "Material & Texture", "Size", "Part-Whole"]
    
    print(f"Starting compound image generation...")
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    print(f"Mode: Creating {args.max_compounds} balanced compound images per category")
    print(f"      All single images will be copied")
    
    for category in categories:
        process_category(category, input_base, output_base, args.max_compounds)
    
    print("\n✓ All images created successfully!")

if __name__ == "__main__":
    main()