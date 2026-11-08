"""Fixtures partagées."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES_DIR = ROOT / "echantillons_20_photos"


@pytest.fixture(scope="session")
def sample_images() -> list[Path]:
    images = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not images:
        pytest.skip("échantillons absents")
    return images


@pytest.fixture(scope="session")
def real_sample(sample_images: list[Path]) -> Path:
    for path in sample_images:
        if "_REAL_" in path.name:
            return path
    pytest.skip("aucun échantillon REAL")


@pytest.fixture(scope="session")
def fake_sample(sample_images: list[Path]) -> Path:
    for path in sample_images:
        if "_FAKE_" in path.name:
            return path
    pytest.skip("aucun échantillon FAKE")
