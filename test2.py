from transformers import CLIPTextModel, CLIPTokenizer

text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="text_encoder")

print(text_encoder.forward.__doc__)
