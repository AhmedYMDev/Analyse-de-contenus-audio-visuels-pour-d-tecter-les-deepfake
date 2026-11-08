"""Tests du détecteur de visages et du recadrage."""

from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("cv2")

from deepfake.face import FaceBox, FaceDetector, crop_face, largest_face  # noqa: E402
from deepfake.preprocess import open_image, to_rgb  # noqa: E402


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    instance = FaceDetector()
    instance.load()
    return instance


def test_detects_a_face_in_a_real_sample(detector, real_sample):
    faces = detector.detect(to_rgb(open_image(real_sample)))
    assert faces, "aucun visage détecté sur un crop de visage Celeb-DF"
    assert faces[0].score > 0.5
    assert faces[0].width > 8 and faces[0].height > 8


def test_detects_a_face_in_a_fake_sample(detector, fake_sample):
    assert detector.detect(to_rgb(open_image(fake_sample)))


def test_returns_empty_on_flat_image(detector):
    assert detector.detect(Image.new("RGB", (256, 256), (120, 120, 120))) == []


def test_boxes_are_sorted_by_area(detector, sample_images):
    # Sur une image avec un seul visage, le tri est trivial ; on vérifie au moins
    # que la liste est ordonnée et que les boîtes sont dans l'image.
    for path in sample_images[:3]:
        image = to_rgb(open_image(path))
        areas = [box.area for box in detector.detect(image)]
        assert areas == sorted(areas, reverse=True)
        for box in detector.detect(image):
            assert 0 <= box.x < image.width
            assert 0 <= box.y < image.height


def test_crop_is_square_and_inside_the_image():
    image = Image.new("RGB", (200, 150), (0, 0, 0))
    box = FaceBox(x=60, y=40, width=50, height=50, score=0.9)
    crop = crop_face(image, box, margin=0.25)
    assert crop.width == crop.height
    assert crop.width <= 200
    assert crop.height <= 150


def test_crop_is_clamped_at_the_border():
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    box = FaceBox(x=0, y=0, width=40, height=40, score=0.9)
    crop = crop_face(image, box, margin=0.5)
    # côté = 40 × (1 + 2×0.5) = 80 px, borné à l'image (coin haut-gauche)
    assert crop.size == (80, 80)
    assert crop.width <= image.width and crop.height <= image.height


def test_largest_face_reports_count(detector, real_sample):
    face, box, count = largest_face(to_rgb(open_image(real_sample)), detector)
    assert face is not None
    assert box is not None
    assert count >= 1


def test_largest_face_returns_none_without_face(detector):
    face, box, count = largest_face(Image.new("RGB", (256, 256), (30, 30, 30)), detector)
    assert face is None and box is None and count == 0
