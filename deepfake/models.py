"""Construction et chargement des modèles, avec garde-fous de déterminisme.

Deux pièges sont traités ici :

1. ``cnn_simple.keras`` a été exporté avec ses couches d'augmentation **dans le
   graphe**. À l'inférence, elles continuaient de s'appliquer : la même image
   donnait deux verdicts différents. ``strip_random_layers`` reconstruit un
   graphe déterministe à poids identiques.
2. Un checkpoint incompatible était chargé avec ``strict=False``, donc en
   silence. Ici, toute clé manquante ou en trop lève une erreur explicite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import DEFAULT_IMG_SIZE

#: Couches stochastiques à retirer avant l'inférence.
RANDOM_LAYER_NAMES = (
    "RandomFlip",
    "RandomRotation",
    "RandomZoom",
    "RandomTranslation",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomErasing",
    "Dropout",
    "SpatialDropout2D",
    "GaussianNoise",
    "GaussianDropout",
)


def build_resnet18(num_classes: int = 1):
    """ResNet-18 avec la même tête que celle utilisée à l'entraînement."""
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes))
    return model


def load_resnet18(path: Path | str, device: Any = "cpu"):
    """Charge un checkpoint PyTorch en mode strict et en lecture sûre."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint ResNet-18 absent : {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Format de checkpoint inattendu dans {path}")

    model = build_resnet18()
    # strict=True : une tête différente ou un backbone différent échoue bruyamment.
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _keras():
    """Import paresseux de Keras (TensorFlow reste une dépendance optionnelle)."""
    try:
        import keras
        from keras import layers as keras_layers
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "TensorFlow/Keras n'est pas installé : le modèle CNN Simple est indisponible."
        ) from exc
    return keras, keras_layers


def strip_random_layers(model):
    """Reconstruit le modèle sans couches aléatoires, en réutilisant les poids.

    Renvoie ``(modele_deterministe, [noms_des_couches_retirees])``.
    """
    keras, keras_layers = _keras()

    removable = tuple(
        getattr(keras_layers, name) for name in RANDOM_LAYER_NAMES if hasattr(keras_layers, name)
    )
    removed: list[str] = []

    inputs = keras.Input(shape=model.input_shape[1:])
    outputs = inputs
    for layer in model.layers[1:]:  # [0] = InputLayer
        if isinstance(layer, removable):
            removed.append(layer.__class__.__name__)
            continue
        outputs = layer(outputs)

    return keras.Model(inputs, outputs, name=f"{model.name}_inference"), removed


def load_cnn(path: Path | str, deterministic: bool = True):
    """Charge le CNN Keras ; retire par défaut les couches stochastiques."""
    keras, _ = _keras()

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle CNN absent : {path}")

    model = keras.models.load_model(path)
    removed: list[str] = []
    if deterministic:
        model, removed = strip_random_layers(model)
    return model, removed


def check_determinism(predict: Callable[[np.ndarray], np.ndarray], shape=(2, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, 3)) -> bool:
    """Vérifie que deux appels identiques donnent exactement les mêmes sorties.

    La sonde est un motif aléatoire **fixe** (et non des zéros) : sur une image
    constante, une rotation ou un changement de luminosité ne modifierait rien
    et le test ne détecterait pas les couches stochastiques.
    """
    rng = np.random.default_rng(0)
    probe = rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
    first = np.asarray(predict(probe), dtype=np.float32)
    second = np.asarray(predict(probe), dtype=np.float32)
    return bool(np.array_equal(first, second))
