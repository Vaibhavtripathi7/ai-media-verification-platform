import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

MAX_ASPECT = 8
FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "AVIF")

# Pillow refuses anything over about 179 MP at open(), before draft() has had a
# chance to shrink it, which would turn away 200 MP phone photos. decode() checks
# the size that will actually be decoded instead.
Image.MAX_IMAGE_PIXELS = None


class InvalidImage(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class Preprocessor:
    """Shortest-edge resize, center crop and normalize, read from the model's
    preprocessor_config.json so served inputs match what the model was trained on."""

    shortest_edge: int
    crop_size: int
    mean: np.ndarray
    std: np.ndarray
    resample: Image.Resampling
    max_pixels: int

    @classmethod
    def from_config(cls, config: dict, max_pixels: int) -> "Preprocessor":
        if not (config.get("do_resize") and config.get("do_center_crop")):
            raise ValueError("expected a resize + center crop preprocessor config")
        return cls(
            shortest_edge=config["size"]["shortest_edge"],
            crop_size=config["crop_size"],
            mean=np.asarray(config["image_mean"], dtype=np.float32).reshape(3, 1, 1),
            std=np.asarray(config["image_std"], dtype=np.float32).reshape(3, 1, 1),
            resample=Image.Resampling(config["resample"]),
            max_pixels=max_pixels,
        )

    def decode(self, data: bytes, draft: bool = True) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(data), formats=FORMATS)
        except OSError as exc:
            raise InvalidImage("not a JPEG, PNG, WebP, GIF or AVIF image") from exc

        if draft and image.format == "JPEG":
            # libjpeg can decode at 1/2, 1/4 or 1/8 scale. Asking for at least the
            # resize target keeps the final resize below doing the real work.
            image.draft("RGB", (self.shortest_edge, self.shortest_edge))

        # The limit is on what will actually be decoded, so a 50 MP phone photo that
        # draft() shrinks is fine while a 50 MP PNG is not. Nothing is decoded yet.
        width, height = image.size
        if width * height > self.max_pixels:
            raise ImageTooLarge(f"{width}x{height} is over {self.max_pixels} pixels")

        try:
            return image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise InvalidImage("image data is truncated or corrupt") from exc

    def __call__(self, image: Image.Image) -> np.ndarray:
        # Only the middle of the image reaches the model. A long, thin upload such as
        # 20 x 2,000,000 would otherwise be resized into a strip billions of pixels
        # long just to have most of it cropped away, so trim the long side first.
        # Nothing in the evaluation data is past 8:1, so those results still hold.
        width, height = image.size
        keep = MAX_ASPECT * min(width, height)
        if width > keep:
            left = (width - keep) // 2
            image = image.crop((left, 0, left + keep, height))
        elif height > keep:
            top = (height - keep) // 2
            image = image.crop((0, top, width, top + keep))

        width, height = image.size
        if width <= height:
            size = (self.shortest_edge, int(self.shortest_edge * height / width))
        else:
            size = (int(self.shortest_edge * width / height), self.shortest_edge)
        image = image.resize(size, self.resample)

        left = (image.width - self.crop_size) // 2
        top = (image.height - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))

        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (pixels - self.mean) / self.std
