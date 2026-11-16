"""Detectors under evaluation. Each one maps raw image bytes to the log-odds that the
image is AI-generated, so scores keep their full resolution near 0 and 1."""

import io
from collections.abc import Callable

from app.config import Settings
from app.detector import Detector
from app.weights import download

Score = Callable[[bytes], float]


def served(draft: bool = True) -> Score:
    """Exactly what the service runs: same decode, preprocessing and ONNX session."""
    settings = Settings()
    detector = Detector(
        download(settings), settings.model_file, settings.threads, settings.max_pixels
    )

    def score(data: bytes) -> float:
        pixels = detector.preprocess(detector.preprocess.decode(data, draft=draft))
        return float(detector.predict(pixels[None])[0])

    return score


def huggingface(repo: str, revision: str, ai_label: str) -> Score:
    """A transformers image classifier run the way its model card shows."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(Settings.threads)
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForImageClassification.from_pretrained(repo, revision=revision).eval()
    ai_index = model.config.label2id[ai_label]

    @torch.inference_mode()
    def score(data: bytes) -> float:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        logits = model(**processor(image, return_tensors="pt")).logits[0]
        return float(logits[ai_index] - logits[1 - ai_index])

    return score


CANDIDATES: dict[str, Callable[[], Score]] = {
    "communityforensics-vit": served,
    "communityforensics-vit-full-decode": lambda: served(draft=False),
    "sdxl-detector": lambda: huggingface(
        "Organika/sdxl-detector", "b37fede8562cb72b89ec201c0987f96ba21b518a", "artificial"
    ),
}
