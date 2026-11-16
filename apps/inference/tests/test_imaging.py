import pytest

from app.imaging import ImageTooLarge, InvalidImage, Preprocessor

from .support import PREPROCESSOR_CONFIG, encode


@pytest.fixture
def preprocess() -> Preprocessor:
    return Preprocessor.from_config(PREPROCESSOR_CONFIG, max_pixels=1_000_000)


@pytest.mark.parametrize("size", [(64, 64), (300, 50), (50, 300), (41, 40)])
def test_output_is_a_normalized_crop(preprocess, size):
    pixels = preprocess(preprocess.decode(encode(size)))
    assert pixels.shape == (3, 32, 32)
    assert pixels.dtype == "float32"
    # (120, 40, 200) / 255 normalized with mean 0.5, std 0.5
    assert pixels[:, 16, 16] == pytest.approx([-0.0588, -0.6863, 0.5686], abs=1e-3)


def test_large_jpeg_is_draft_decoded_but_still_covers_the_crop(preprocess):
    image = preprocess.decode(encode((1000, 800), "JPEG"))
    assert min(image.size) >= 40
    assert image.size[0] < 1000
    assert preprocess(image).shape == (3, 32, 32)


def test_rejects_bytes_that_are_not_an_image(preprocess):
    with pytest.raises(InvalidImage):
        preprocess.decode(b"definitely not a png")


def test_rejects_truncated_image(preprocess):
    with pytest.raises(InvalidImage):
        preprocess.decode(encode((200, 200), "JPEG")[:300])


def test_checks_pixel_count_before_decoding(preprocess):
    with pytest.raises(ImageTooLarge):
        preprocess.decode(encode((1001, 1000)))


def test_pixel_limit_applies_to_the_size_a_jpeg_decodes_to(preprocess):
    big_jpeg = encode((1200, 1000), "JPEG")
    assert min(preprocess.decode(big_jpeg).size) >= 40
    with pytest.raises(ImageTooLarge):
        preprocess.decode(big_jpeg, draft=False)
