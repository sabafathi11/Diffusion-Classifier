from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
import numpy as np

# Load your existing image
image_path = "dog1.jpg"
image_pil = Image.open(image_path).convert("RGB").resize((1024, 1024))

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
).to(device)


prompt = "bird horse cat deer frog dog truck airplane automobile ship"
gen = set_seed(0)

classes = prompt.split()

for label in classes :
    print(label,end='\n\n')
    with torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(
                prompt=prompt,
                image=image_pil,
                strength=0.03,      # tiny value to avoid changes but keep pipeline working
                num_inference_steps=150,
                generator=gen
            )

            heat_map = tc.compute_global_heat_map()
            heat_map = heat_map.compute_word_heat_map(label)

            # heat_map.plot_overlay(out.images[0])
            # plt.savefig(f'heat_map_{label}1.png')

            
            heat_arr = np.array(heat_map.heatmap.cpu()) # shape: (H, W), values in [0, 1]

            print(heat_arr.shape)
            print(heat_arr.)

            # total_attention = heat_arr.sum()
            # print(f"Total attention value for: {total_attention}")


            # threshold = 0.15
            # attended_pixels = (heat_arr >= threshold).sum()
            # total_pixels = heat_arr.size
            # attended_percentage = attended_pixels / total_pixels * 100
            # print(f"Attended pixels for {label}: {attended_pixels} / {total_pixels} ({attended_percentage:.2f}%)")


            print('\n')

