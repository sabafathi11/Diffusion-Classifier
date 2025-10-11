from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
import numpy as np

# Load your existing image
image_path = "Banana_433.jpg"
image_pil = Image.open(image_path).convert("RGB").resize((1024, 1024))

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'
custom_cache = "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/SD"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
    cache_dir=custom_cache,
).to(device)

gen = set_seed(0)
prompt = "a yellow banana"
label = "banana"


with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(
            prompt=prompt,
            image=image_pil,
            strength=0.03,      # tiny value to avoid changes but keep pipeline working
            num_inference_steps=50,
            generator=gen
        )

        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(label)

        heat_map.plot_overlay(out.images[0])
        plt.savefig(f'heat_map_{label}1.png')
