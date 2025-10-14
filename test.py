import matplotlib.pyplot as plt
from datasets import load_dataset

# Load datasets
comco = load_dataset("clip-oscope/simco-comco", data_dir="ComCo")
simco = load_dataset("clip-oscope/simco-comco", data_dir="SimCo")

# Function to display images from a dataset
def show_images(dataset, dataset_name, num_images=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{dataset_name} Sample Images', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            # Get the image
            image = dataset['train'][idx]['image']
            
            # Display the image
            ax.imshow(image)
            ax.set_title(f'Index: {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_samples.png')

# Show ComCo samples
show_images(comco, 'ComCo', num_images=6)

# Show SimCo samples
show_images(simco, 'SimCo', num_images=6)

# To see more details about a specific image:
print("\nComCo first image details:")
print(f"Type: {type(comco['train'][0]['image'])}")
print(f"Size: {comco['train'][0]['image'].size}")
print(f"Mode: {comco['train'][0]['image'].mode}")

# To access image file names if they exist in the dataset metadata:
print("\nDataset features:")
print(comco['train'].features)