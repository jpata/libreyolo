import numpy as np
from PIL import Image

from ...utils.image_loader import ImageLoader


def preprocess_image(image, input_size: int, color_format: str = "auto"):
    img = ImageLoader.load(image, color_format=color_format)
    pil_img = img.convert("RGB")
    orig_w, orig_h = pil_img.size

    resized = pil_img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
    tensor = np.expand_dims(chw, axis=0)

    scale = float(input_size) / max(orig_w, orig_h)
    return tensor, pil_img, (orig_w, orig_h), scale