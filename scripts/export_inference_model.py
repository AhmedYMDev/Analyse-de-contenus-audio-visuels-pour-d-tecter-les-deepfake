#!/usr/bin/env python3
"""Réexporte le CNN Simple **sans** ses couches aléatoires.

Le fichier ``cnn_simple.keras`` du dépôt contient RandomFlip / RandomRotation /
RandomBrightness / RandomContrast / Dropout dans son graphe : à l'inférence, ils
continuaient de s'appliquer et la même image pouvait changer de verdict.

Ce script produit un artefact propre et vérifie qu'il est déterministe.

    python scripts/export_inference_model.py
    python scripts/export_inference_model.py --input cnn_simple.keras --output cnn_simple_inference.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepfake.config import DEFAULT_IMG_SIZE  # noqa: E402
from deepfake.models import check_determinism, load_cnn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cnn_simple.keras")
    parser.add_argument("--output", type=Path, default=ROOT / "cnn_simple_inference.keras")
    parser.add_argument("--size", type=int, default=DEFAULT_IMG_SIZE)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Modèle introuvable : {args.input}", file=sys.stderr)
        return 1

    model, removed = load_cnn(args.input, deterministic=True)
    print(f"Couches stochastiques retirées : {', '.join(removed) or 'aucune'}")

    probe = np.zeros((4, args.size, args.size, 3), dtype=np.float32)
    deterministic = check_determinism(lambda batch: model.predict(batch, verbose=0))
    print(f"Déterminisme (4 appels identiques) : {'OK' if deterministic else 'ÉCHEC'}")
    if not deterministic:
        print("Le modèle n'est pas déterministe : artefact non exporté.", file=sys.stderr)
        return 1

    predictions = model.predict(probe, verbose=0)
    print(f"Forme de sortie : {predictions.shape} — exemple : {predictions.ravel()[:4]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Artefact écrit : {args.output}")
    print("Pensez à le référencer dans model_card.json et à relancer l'application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
