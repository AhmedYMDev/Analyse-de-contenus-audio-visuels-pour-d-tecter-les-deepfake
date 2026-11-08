"""Configuration partagée : chemins et « carte de modèle ».

La carte de modèle (``model_card.json``) est le contrat entre l'entraînement et
le déploiement : taille d'entrée, normalisation, sens des labels, seuil de
décision. L'écrire une fois et la lire partout évite les décalages silencieux
entre le notebook et l'application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CARD_PATH = BASE_DIR / "model_card.json"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# Valeurs de repli : elles doivent rester alignées sur l'entraînement.
DEFAULT_IMG_SIZE = 128
DEFAULT_MEAN = (0.485, 0.456, 0.406)  # ImageNet
DEFAULT_STD = (0.229, 0.224, 0.225)   # ImageNet
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ModelCard:
    """Paramètres de déploiement lus depuis ``model_card.json``."""

    img_size: int = DEFAULT_IMG_SIZE
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    threshold: float = DEFAULT_THRESHOLD
    class_1_is_fake: bool = True
    face_crop_margin: float = 0.25
    #: taille maximale acceptée en entrée, en pixels (garde-fou « decompression bomb »)
    max_image_pixels: int = 40_000_000
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> tuple[str, str]:
        return ("REAL", "FAKE") if self.class_1_is_fake else ("FAKE", "REAL")


def _as_float_triplet(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        triplet = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return fallback
    return triplet if len(triplet) == 3 else fallback  # type: ignore[return-value]


def load_model_card(path: Path | str = MODEL_CARD_PATH) -> ModelCard:
    """Charge la carte de modèle, en retombant sur des valeurs sûres si absente."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} n'est pas un JSON valide : {exc}") from exc

    normalization = raw.get("normalization", {}) or {}
    return ModelCard(
        img_size=int(raw.get("img_size", DEFAULT_IMG_SIZE)),
        mean=_as_float_triplet(normalization.get("mean"), DEFAULT_MEAN),
        std=_as_float_triplet(normalization.get("std"), DEFAULT_STD),
        threshold=float(raw.get("decision_threshold", DEFAULT_THRESHOLD)),
        class_1_is_fake=bool(raw.get("class_1_is_fake", True)),
        face_crop_margin=float(raw.get("face_crop_margin", 0.25)),
        raw=raw,
    )
