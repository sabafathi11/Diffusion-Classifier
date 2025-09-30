import os
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

# --- Model setup ---
torch.set_float32_matmul_precision("high")

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to("cuda")

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Processing functions ---
def process(image: Image.Image) -> Image.Image:
    """Remove background from a PIL image."""
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def process_file(filepath: str, save=True) -> str:
    """
    Load an image from local disk, remove background, save result.
    
    Args:
        filepath (str): Path to image file
        save (bool): Whether to save output
    
    Returns:
        str: Path to saved image (if save=True) or just processed image object
    """
    im = Image.open(filepath).convert("RGB")
    transparent = process(im)
    if save:
        out_path = os.path.splitext(filepath)[0] + "_transparent.png"
        transparent.save(out_path)
        return out_path
    return transparent
# Example usage:
result_path = process_file("image.jpg")
print("Saved to:", result_path)
