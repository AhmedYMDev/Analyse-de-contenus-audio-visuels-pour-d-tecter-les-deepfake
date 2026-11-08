#!/usr/bin/env python3
"""Télécharge le détecteur de visages YuNet utilisé par l'application.

Le fichier n'est pas versionné par confort ; ce script le récupère depuis le
dépôt officiel OpenCV Zoo et vérifie son empreinte SHA-256.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.is_file() and sha256(TARGET) == EXPECTED_SHA256:
        print(f"Déjà présent et valide : {TARGET}")
        return 0

    print(f"Téléchargement de YuNet depuis {URL}")
    try:
        urllib.request.urlretrieve(URL, TARGET)
    except Exception as exc:  # noqa: BLE001
        print(f"Échec du téléchargement : {exc}", file=sys.stderr)
        return 1

    actual = sha256(TARGET)
    if actual != EXPECTED_SHA256:
        print(
            f"Empreinte inattendue : {actual}\nAttendu : {EXPECTED_SHA256}",
            file=sys.stderr,
        )
        return 1

    print(f"OK : {TARGET} ({TARGET.stat().st_size / 1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
