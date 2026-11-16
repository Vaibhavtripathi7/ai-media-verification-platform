"""OpenFakeTiny (https://huggingface.co/datasets/ComplexDataLab/OpenFakeTiny).

Fake images come from generators released in 2024-2026, real ones from ImageNet and
DOCCI. The reddit split is in-the-wild: images and video frames posted to Reddit,
labelled by the dataset authors.
"""

import io
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "ComplexDataLab/OpenFakeTiny"
REVISION = "9a7f5a7b37490f1b7d557c285eef52b09a90d0c2"

SPLITS = {
    "validation": [
        "core/validation-00000-of-00002.parquet",
        "core/validation-00001-of-00002.parquet",
    ],
    "test": ["core/test-00000-of-00001.parquet"],
    "reddit": ["reddit/test-00000-of-00001.parquet"],
}


@dataclass(frozen=True)
class Sample:
    id: str
    data: bytes
    is_ai: bool
    source: str


def samples(split: str) -> Iterator[Sample]:
    index = 0
    for name in SPLITS[split]:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=32, columns=["image", "label", "model"]
        ):
            for row in batch.to_pylist():
                yield Sample(
                    id=f"{split}-{index}",
                    data=row["image"]["bytes"],
                    is_ai=row["label"] == "fake",
                    source=row["model"],
                )
                index += 1


def recompress(data: bytes, long_side: int = 1080, quality: int = 75) -> bytes:
    """Roughly what a chat app or social feed does to an upload."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=quality)
    return out.getvalue()
