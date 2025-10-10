import torch
from torchvision import transforms as torch_transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from PIL import Image

# Helper to ensure RGB
def _convert_image_to_rgb(image):
    return image.convert("RGB")

# Define transform
size = 512
interpolation = InterpolationMode.BICUBIC


# transform = transforms.Compose([
#     transforms.Resize(512),
#     transforms.CenterCrop(512),
#     transforms.ColorJitter(
#         brightness=0.5,   # randomly change brightness
#         contrast=0.5,     # randomly change contrast
#         saturation=0.5,   # randomly change saturation
#         hue=0.2           # randomly change hue (color shift)
#     ),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

transform = torch_transforms.Compose([
    torch_transforms.Resize(size, interpolation=interpolation),
    torch_transforms.CenterCrop(size),
    _convert_image_to_rgb,
    torch_transforms.Lambda(lambda img: torch_transforms.functional.adjust_hue(img, -0.5)),  # Major color shift
    torch_transforms.Lambda(lambda img: torch_transforms.functional.adjust_saturation(img, 2.5)),  # Hyper-saturated
    torch_transforms.Lambda(lambda img: torch_transforms.functional.adjust_contrast(img, 1.8)),  # Punchy contrast
    torch_transforms.Lambda(lambda img: torch_transforms.functional.adjust_brightness(img, 1)),  # Slight brightness boost
    torch_transforms.ToTensor(),
    torch_transforms.Normalize([0.5], [0.5])
])

# Load image
input_path = "image.jpg"   # change this to your image path
output_path = "output3.png"

image = Image.open(input_path)

# Apply transform
img_tensor = transform(image)

# Undo normalization for saving (bring values back to [0,1])
img_to_save = img_tensor * 0.5 + 0.5  

# Save transformed image
save_image(img_to_save, output_path)

print(f"Transformed image saved at: {output_path}")


# 999 timestep 
# background remove
# lemon testing in test
