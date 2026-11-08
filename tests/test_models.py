"""Tests de chargement des modèles et du garde-fou de déterminisme.

Marqués ``models`` : ils nécessitent PyTorch / TensorFlow et les artefacts du
dépôt. Lancer avec ``pytest -m models`` (ou sans filtre).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepfake.models import (  # noqa: E402
    build_resnet18,
    check_determinism,
    load_cnn,
    load_resnet18,
)

pytestmark = pytest.mark.models
ROOT = Path(__file__).resolve().parent.parent


def test_build_resnet18_head_matches_checkpoint():
    model = build_resnet18()
    assert model.fc[1].in_features == 512
    assert model.fc[1].out_features == 1


def test_load_resnet18_strict_and_predicts():
    model = load_resnet18(ROOT / "resnet18.pt", "cpu")
    tensor = torch.zeros(1, 3, 128, 128)
    with torch.inference_mode():
        logits = model(tensor)
    assert logits.shape == (1, 1)
    assert np.isfinite(logits.item())


def test_load_resnet18_missing_file():
    with pytest.raises(FileNotFoundError):
        load_resnet18(ROOT / "does-not-exist.pt", "cpu")


def test_cnn_is_deterministic_after_stripping():
    pytest.importorskip("tensorflow")
    model, removed = load_cnn(ROOT / "cnn_simple.keras", deterministic=True)
    assert "RandomFlip" in removed
    assert "Dropout" in removed
    assert check_determinism(lambda batch: model.predict(batch, verbose=0))


def test_raw_cnn_is_flagged_as_non_deterministic():
    """Régression : le garde-fou doit détecter l'artefact stochastique d'origine."""
    pytest.importorskip("tensorflow")
    raw, removed = load_cnn(ROOT / "cnn_simple.keras", deterministic=False)
    assert removed == []
    assert not check_determinism(lambda batch: raw.predict(batch, verbose=0))
