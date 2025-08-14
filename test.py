from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
import numpy as np


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

def gaussian_center_weights(H, W, sigma=0.35):
    # sigma is relative to image size (0..1): larger = wider center
    ys, xs = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    sy, sx = sigma * H, sigma * W
    w = np.exp(-(((ys - cy)**2) / (2 * sy**2) + ((xs - cx)**2) / (2 * sx**2)))
    return w

def gaussian_center_score(hm, sigma=0.35, top_percent=0.1):
    H, W = hm.shape
    w = gaussian_center_weights(H, W, sigma)
    weighted = hm * w

    if top_percent is None:
        # weighted mean
        return weighted.sum() / (w.sum() + 1e-8)

    flat = weighted.ravel()
    thr = np.percentile(flat, 100 - top_percent * 100)
    return flat[flat >= thr].mean()

scores = {}

for label in classes:
    with torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(
                prompt=prompt,
                image=image_pil,
                strength=0.02,
                num_inference_steps=100,
                generator=gen
            )

            heat_map = tc.compute_global_heat_map().compute_word_heat_map(label)

            # Optional: save overlay
            # heat_map.plot_overlay(out.images[0])
            # plt.savefig(f'heat_map_{label}1.png')

            # Tensor -> numpy
            hm = np.array(heat_map.heatmap.cpu())

            # Per-label robust normalization (optional but helps comparability)
            lo = np.percentile(hm, 1)
            hi = np.percentile(hm, 99)
            hm = np.clip((hm - lo) / (hi - lo + 1e-8), 0, 1)

            # === Choose ONE of these scoring functions ===
            #score = center_crop_score(hm, center_frac=0.60, top_percent=0.10)
            score = gaussian_center_score(hm, sigma=0.15, top_percent=0.05)

            scores[label] = float(score)
            print(f"{label:>10s}  score: {score:.6f}")

# Pick the predicted label
pred = max(scores, key=scores.get)
print("\nPredicted:", pred)
print("Ranking:", sorted(scores.items(), key=lambda x: x[1], reverse=True))



""""
methods to try:

max
min
mean 
count pixels
threshold with min and mean
more weight to center and bigger attention

"""