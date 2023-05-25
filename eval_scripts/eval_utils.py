import torch
from PIL import Image


def load_image(image, image_processor):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
        image = image_processor(raw_image).unsqueeze(0)
    elif isinstance(image, Image.Image):
        raw_image = image
        image = image_processor(raw_image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
    return image
