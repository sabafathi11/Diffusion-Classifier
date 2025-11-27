import os
import shutil
from pathlib import Path
from collections import Counter
import csv

def load_class_names(labels_csv_path):
    """Load class names from labels.csv file."""
    class_names = {}
    with open(labels_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                class_id = row[0].strip()
                class_name = row[1].strip()
                class_names[class_id] = class_name
    return class_names

def find_common_classes(base_dir, n=10):
    """
    Find common classes across all folders.
    
    Args:
        base_dir: Path to the base directory
        n: Number of top classes to consider
    
    Returns:
        Set of common class IDs and dict of all folder names
    """
    base_path = Path(base_dir)
    
    # Skip these files/folders
    skip = {'labels.csv', '.venv', 'venv', '__pycache__', '.git', '.idea'}
    
    # Get all subdirectories
    main_folders = [f for f in base_path.iterdir() 
                   if f.is_dir() and f.name not in skip]
    
    top_classes_sets = {}
    
    for main_folder in main_folders:
        class_counts = {}
        
        # Get all class folders
        class_folders = [f for f in main_folder.iterdir() if f.is_dir()]
        
        for class_folder in class_folders:
            # Count image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            image_count = sum(1 for f in class_folder.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions)
            
            if image_count > 0:
                class_counts[class_folder.name] = image_count
        
        # Get top N classes
        top_classes = Counter(class_counts).most_common(n)
        top_classes_sets[main_folder.name] = set(class_id for class_id, _ in top_classes)
    
    # Find intersection
    if top_classes_sets:
        common_classes = set.intersection(*top_classes_sets.values())
    else:
        common_classes = set()
    
    return common_classes, [f.name for f in main_folders]

def copy_common_classes(source_dir, dest_dir, common_classes, folder_names):
    """
    Copy common classes from source to destination maintaining structure.
    
    Args:
        source_dir: Source base directory
        dest_dir: Destination base directory
        common_classes: Set of common class IDs to copy
        folder_names: List of folder names to process
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy labels.csv
    labels_source = source_path / 'labels.csv'
    labels_dest = dest_path / 'labels.csv'
    if labels_source.exists():
        shutil.copy2(labels_source, labels_dest)
        print(f"✓ Copied labels.csv")
    
    total_folders_copied = 0
    total_images_copied = 0
    
    # Process each main folder
    for folder_name in folder_names:
        print(f"\n{'='*70}")
        print(f"Processing: {folder_name}")
        print('='*70)
        
        source_folder = source_path / folder_name
        dest_folder = dest_path / folder_name
        
        # Create destination folder
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        folder_images = 0
        
        # Copy each common class
        for class_id in sorted(common_classes):
            source_class_folder = source_folder / class_id
            dest_class_folder = dest_folder / class_id
            
            if source_class_folder.exists() and source_class_folder.is_dir():
                # Create destination class folder
                dest_class_folder.mkdir(parents=True, exist_ok=True)
                
                # Copy all images
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
                images_copied = 0
                
                for file in source_class_folder.iterdir():
                    if file.is_file() and file.suffix.lower() in image_extensions:
                        shutil.copy2(file, dest_class_folder / file.name)
                        images_copied += 1
                
                if images_copied > 0:
                    print(f"  ✓ {class_id}: copied {images_copied} images")
                    folder_images += images_copied
                    total_folders_copied += 1
        
        print(f"\nTotal images copied from {folder_name}: {folder_images}")
        total_images_copied += folder_images
    
    return total_folders_copied, total_images_copied

def main(source_dir, dest_dir, n_top_classes=10):
    """
    Main function to find and copy common classes.
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
        n_top_classes: Number of top classes to consider
    """
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Finding top {n_top_classes} classes...\n")
    
    # Load class names for reporting
    labels_path = Path(source_dir) / 'labels.csv'
    class_names = load_class_names(labels_path)
    
    # Find common classes
    common_classes, folder_names = find_common_classes(source_dir, n_top_classes)
    
    print(f"Found {len(common_classes)} common classes across all folders:")
    for class_id in sorted(common_classes):
        class_name = class_names.get(class_id, "Unknown")
        print(f"  • {class_id} | {class_name}")
    
    if not common_classes:
        print("\nNo common classes found. Nothing to copy.")
        return
    
    # Ask for confirmation
    response = input(f"\nProceed with copying to '{dest_dir}'? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    print("\nStarting copy operation...")
    
    # Copy the classes
    folders_copied, images_copied = copy_common_classes(
        source_dir, dest_dir, common_classes, folder_names
    )
    
    print(f"\n{'='*70}")
    print("COPY COMPLETE")
    print('='*70)
    print(f"Common classes: {len(common_classes)}")
    print(f"Folders processed: {len(folder_names)}")
    print(f"Class folders copied: {folders_copied}")
    print(f"Total images copied: {images_copied}")
    print(f"\nDestination: {dest_dir}")


if __name__ == "__main__":
    # Configuration
    source_directory = "/work/gn21/h62001/Diffusion-Classifier/saba/datasets/imagenet-b"
    destination_directory = "/work/gn21/h62001/Diffusion-Classifier/saba/datasets/imagenet-b-selected"  # Change this to your desired destination
    n_top_classes = 20  # Number of top classes to consider for intersection
    
    main(source_directory, destination_directory, n_top_classes)