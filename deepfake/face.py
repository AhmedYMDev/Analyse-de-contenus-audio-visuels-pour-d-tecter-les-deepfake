"""Détection et recadrage de visage (YuNet, OpenCV).

Les modèles ont appris sur des **crops de visage** 128 × 128. Servir une photo
entière sans recadrage produit des prédictions qui n'ont aucun sens : ce module
est donc la première étape obligatoire de l'inférence.

YuNet (``cv2.FaceDetectorYN``) est un détecteur léger (~230 Ko) qui tourne sur
CPU en quelques millisecondes et ne nécessite aucun service externe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import FACE_MODEL_PATH


class FaceDetectorUnavailable(RuntimeError):
    """OpenCV ou le modèle YuNet est absent."""


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float
    score: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


class FaceDetector:
    """Détecteur de visages thread-safe (l'objet OpenCV ne l'est pas)."""

    def __init__(
        self,
        model_path: Path | str = FACE_MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._detector = None
        self._lock = threading.Lock()
        self._input_size: tuple[int, int] | None = None

    # -- cycle de vie -------------------------------------------------------
    def load(self) -> None:
        if self._detector is not None:
            return
        try:
            import cv2  # import paresseux : OpenCV reste optionnel
        except ImportError as exc:  # pragma: no cover - dépend de l'environnement
            raise FaceDetectorUnavailable(
                "OpenCV n'est pas installé : pip install opencv-python-headless"
            ) from exc

        if not self.model_path.is_file():
            raise FaceDetectorUnavailable(
                f"Modèle de détection absent : {self.model_path}. "
                "Exécutez `python scripts/download_face_model.py`."
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            5000,
        )
        self._cv2 = cv2

    @property
    def available(self) -> bool:
        return self._detector is not None

    # -- inférence ----------------------------------------------------------
    def detect(self, image: Image.Image) -> list[FaceBox]:
        """Détecte les visages d'une image RGB et renvoie les boîtes en pixels."""
        if self._detector is None:
            self.load()

        width, height = image.size
        if width < 16 or height < 16:
            return []

        # YuNet attend du BGR uint8 continu.
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])

        with self._lock:
            if self._input_size != (width, height):
                self._detector.setInputSize((width, height))
                self._input_size = (width, height)
            _, faces = self._detector.detect(bgr)

        if faces is None:
            return []
        boxes = [
            FaceBox(
                x=float(face[0]),
                y=float(face[1]),
                width=float(face[2]),
                height=float(face[3]),
                score=float(face[-1]),
            )
            for face in faces
            if face[2] > 8 and face[3] > 8
        ]
        # Le plus grand visage d'abord : c'est celui qu'on analyse par défaut.
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes


def crop_face(image: Image.Image, box: FaceBox, margin: float = 0.25) -> Image.Image:
    """Recadre autour de la boîte avec une marge relative, en restant dans l'image.

    Le crop est carré (comme à l'entraînement) et les coordonnées sont bornées aux
    dimensions de l'image, donc aucun remplissage noir n'est introduit.
    """
    width, height = image.size
    side = max(box.width, box.height) * (1.0 + 2.0 * margin)
    side = min(side, float(max(width, height)))

    center_x, center_y = box.center
    left = max(0.0, center_x - side / 2.0)
    top = max(0.0, center_y - side / 2.0)
    right = min(float(width), left + side)
    bottom = min(float(height), top + side)

    # Re-cale si le crop a été rogné par un bord.
    left = max(0.0, right - side)
    top = max(0.0, bottom - side)

    return image.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))


def largest_face(image: Image.Image, detector: FaceDetector, margin: float = 0.25):
    """Renvoie ``(crop, boîte, nombre_de_visages)`` ou ``(None, None, n)``."""
    faces = detector.detect(image)
    if not faces:
        return None, None, 0
    return crop_face(image, faces[0], margin), faces[0], len(faces)
