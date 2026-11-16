import io
import threading

import numpy as np
from PIL import Image

from app.imaging import Preprocessor

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


class FakeDetector:
    def __init__(self, logit: float = 1.5, max_pixels: int = 1_000_000):
        self.preprocess = Preprocessor.from_config(PREPROCESSOR_CONFIG, max_pixels)
        self.logit = logit
        self.release = threading.Event()
        self.release.set()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.release.wait(timeout=5)
        return np.full(len(batch), self.logit, dtype=np.float32)


def encode(size: tuple[int, int], fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (120, 40, 200)).save(out, fmt)
    return out.getvalue()
