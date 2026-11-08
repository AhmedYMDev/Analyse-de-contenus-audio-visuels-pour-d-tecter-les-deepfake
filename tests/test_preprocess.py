"""Tests du prétraitement : EXIF, transparence, dimensions, normalisation."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from deepfake.preprocess import (
    ImageDecodeError,
    cnn_batch,
    open_image,
    prepare,
    resnet_tensor,
    to_rgb,
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_alpha_is_composited_on_white():
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    rgb = to_rgb(image)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (255, 255, 255)


def test_palette_with_transparency_is_composited():
    image = Image.new("P", (8, 8), 0)
    image.info["transparency"] = 0
    assert to_rgb(image).mode == "RGB"


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = image.getexif()
    exif[274] = 6  # rotation 90°
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    decoded = to_rgb(open_image(buffer.getvalue()))
    assert decoded.size == (20, 40)


def test_prepare_returns_square_rgb():
    image = Image.new("RGB", (300, 120), (10, 20, 30))
    prepared = prepare(image, 128)
    assert prepared.size == (128, 128)
    assert prepared.mode == "RGB"


def test_resnet_tensor_shape_and_normalization():
    image = Image.new("RGB", (128, 128), (128, 128, 128))
    tensor = resnet_tensor(image, 128, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert tuple(tensor.shape) == (1, 3, 128, 128)
    assert tensor.dtype.is_floating_point
    expected = (128 / 255.0 - 0.485) / 0.229
    assert tensor[0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-5)


def test_cnn_batch_keeps_pixel_scale():
    image = Image.new("RGB", (128, 128), (255, 128, 0))
    batch = cnn_batch(image, 128)
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert batch[0, 0, 0].tolist() == [255.0, 128.0, 0.0]


def test_open_image_accepts_bytes_and_path(tmp_path):
    image = Image.new("RGB", (16, 16), (1, 2, 3))
    path = tmp_path / "tiny.png"
    image.save(path)

    assert open_image(path).size == (16, 16)
    assert open_image(_png_bytes(image)).size == (16, 16)


def test_open_image_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        open_image(b"ceci n'est pas une image")


def test_decompression_bomb_guard_is_active():
    assert Image.MAX_IMAGE_PIXELS == 40_000_000
