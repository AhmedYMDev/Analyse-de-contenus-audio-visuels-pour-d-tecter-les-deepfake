"""Prétraitement des images, partagé par le notebook, l'application et les scripts.

Règles (identiques à l'entraînement) :

* correction EXIF, composition du canal alpha sur fond blanc, conversion RGB ;
* redimensionnement en 128 × 128 avec une interpolation **bilinéaire explicite** ;
* ResNet-18 : tenseur normalisé ImageNet ; CNN Simple : flottants 0–255
  (le ``Rescaling(1./255)`` est dans le modèle Keras).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from .config import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD

# Garde-fou contre les images « decompression bomb » : 40 Mpx ≈ 8000 × 5000.
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

#: Interpolation utilisée à l'entraînement comme à l'inférence.
RESAMPLE = Image.Resampling.BILINEAR


class ImageDecodeError(ValueError):
    """Le fichier n'est pas une image exploitable."""


def open_image(source: bytes | str | Path) -> Image.Image:
    """Décode une image depuis des octets ou un chemin, et la charge en mémoire."""
    try:
        if isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(source)))
        else:
            image = Image.open(str(source))
        image.load()  # force le décodage : les erreurs apparaissent ici, pas plus tard
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError(f"Image illisible : {exc}") from exc


def to_rgb(image: Image.Image) -> Image.Image:
    """EXIF, transparence et modes exotiques → RGB 8 bits."""
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # EXIF corrompu : on ignore plutôt que d'échouer
        pass

    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    return image.convert("RGB")


def resize_face(image: Image.Image, size: int = DEFAULT_IMG_SIZE) -> Image.Image:
    """Redimensionne en carré ``size × size`` (bilinéaire, comme à l'entraînement)."""
    if image.size == (size, size):
        return image
    return image.resize((size, size), RESAMPLE)


def prepare(image: Image.Image, size: int = DEFAULT_IMG_SIZE) -> Image.Image:
    """Pipeline complet RGB → 128 × 128, sans normalisation."""
    return resize_face(to_rgb(image), size)


def resnet_tensor(
    image: Image.Image,
    size: int = DEFAULT_IMG_SIZE,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    """Tenseur ``[1, 3, S, S]`` normalisé ImageNet pour ResNet-18."""
    import torch

    array = np.asarray(prepare(image, size), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def cnn_batch(image: Image.Image, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Lot ``[1, S, S, 3]`` en flottants 0–255 (le modèle Keras normalise seul)."""
    return np.asarray(prepare(image, size), dtype=np.float32)[None, ...]
