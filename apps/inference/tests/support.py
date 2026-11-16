import io

from PIL import Image

# Same shape as the real model's preprocessor_config.json, smaller sizes to keep tests fast.
PREPROCESSOR_CONFIG = {
    "do_resize": True,
    "do_center_crop": True,
    "size": {"shortest_edge": 40},
    "crop_size": 32,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 3,
}


def encode(size: tuple[int, int], fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (120, 40, 200)).save(out, fmt)
    return out.getvalue()
