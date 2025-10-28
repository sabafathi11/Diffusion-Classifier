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
        "Ball": "round",
        "Box": "square",
        "Plate": "circular",
        "Clock": "round",
        "Window": "rectangular",
        "Wheel": "circular",
        "Dice": "cubic",
        "Coin": "round",
        "Book": "rectangular",
        "Table": "rectangular"
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
        "Ant": "small",
        "Whale": "huge",
        "Mouse": "small",
        "wind turbine": "tall",
        "Pebble": "tiny",
        "Truck": "large",
        "Key": "small",
        "Building": "tall",
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
    "Shape": [
        {"round", "circular"},
        {"square", "rectangular", "cubic"}
    ],
    "Size": [
        {"big", "large", "huge", "tall"},
        {"small", "tiny"}
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

def get_first_image(obj_folder):
    """Get the first image from an object folder."""
    for file in sorted(os.listdir(obj_folder)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(obj_folder, file)
    return None

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

def process_category(category_name, input_base, output_base, max_compounds=None):
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
    
    # Get all object folders
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
                img_path = get_first_image(obj_path)
                if img_path:
                    objects[matching_key] = {
                        'path': img_path,
                        'attribute': ATTRIBUTES[category_name][matching_key],
                        'name': obj_name
                    }
    
    if len(objects) < 2:
        print(f"Warning: Not enough objects found in '{category_name}'")
        return
    
    # Generate valid pairs
    obj_keys = list(objects.keys())
    valid_pairs = []
    
    for i, obj1_key in enumerate(obj_keys):
        for obj2_key in obj_keys[i+1:]:
            attr1 = objects[obj1_key]['attribute']
            attr2 = objects[obj2_key]['attribute']
            
            # Check if attributes are different enough
            if not are_attributes_similar(attr1, attr2, category_name):
                valid_pairs.append((obj1_key, obj2_key))
    
    # Shuffle and limit pairs
    random.shuffle(valid_pairs)
    if max_compounds:
        valid_pairs = valid_pairs[:max_compounds]
    
    print(f"\n{category_name}: Creating {len(valid_pairs)} compound images")
    
    # Track which single images we've copied
    copied_singles = set()
    
    # Create compound images
    for obj1_key, obj2_key in valid_pairs:
        obj1 = objects[obj1_key]
        obj2 = objects[obj2_key]
        
        # Copy single images if not already copied
        for obj_key, obj in [(obj1_key, obj1), (obj2_key, obj2)]:
            if obj_key not in copied_singles:
                single_filename = f"{obj['attribute']}_{obj['name']}{Path(obj['path']).suffix}"
                single_output = os.path.join(single_dir, single_filename)
                Image.open(obj['path']).save(single_output)
                copied_singles.add(obj_key)
        
        # Create compound image
        compound_filename = f"{obj1['attribute']}_{obj1['name']}_{obj2['attribute']}_{obj2['name']}.png"
        compound_output = os.path.join(compound_dir, compound_filename)
        
        create_compound_image(obj1['path'], obj2['path'], compound_output)
        print(f"  Created: {compound_filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate compound images from object pairs')
    parser.add_argument('--input', default='other-attributes-nobag', 
                        help='Input directory containing category folders')
    parser.add_argument('--output', default='output', 
                        help='Output directory for results')
    parser.add_argument('--max-compounds', type=int, default=None,
                        help='Maximum number of compound images per category')
    
    args = parser.parse_args()
    
    input_base = args.input
    output_base = args.output
    
    # Process each category
    categories = ["Part-Whole", "Shape", "Material & Texture", "Size", "Temperature"]
    
    print(f"Starting compound image generation...")
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    if args.max_compounds:
        print(f"Max compounds per category: {args.max_compounds}")
    
    for category in categories:
        process_category(category, input_base, output_base, args.max_compounds)
    
    print("\n✓ All compound images created successfully!")

if __name__ == "__main__":
    main()