"""Tests de l'API Flask (chargement réel des modèles)."""

from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")

import app as application  # noqa: E402
from PIL import Image  # noqa: E402

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def client():
    application.load_runtime()
    application.app.config.update(TESTING=True)
    return application.app.test_client()


def _upload(client, data: bytes, name: str = "sample.jpg"):
    return client.post(
        "/api/predict",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


def test_health_reports_models(client):
    response = client.get("/health")
    payload = response.get_json()

    assert response.status_code in (200, 503)
    assert "models" in payload and "model_card" in payload
    assert payload["model_card"]["img_size"] == application.CARD.img_size
    if response.status_code == 503:
        pytest.skip(f"chargement partiel : {payload['errors']}")
    assert payload["models"]["face_detector"] is True


def test_index_renders_the_upload_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Run Analysis" in response.data


def test_predict_returns_both_models(client, real_sample):
    response = _upload(client, real_sample.read_bytes())
    payload = response.get_json()

    assert response.status_code == 200, payload
    assert payload["status"] == "success"
    assert set(payload["models"]) == {"resnet18", "cnn_simple"}
    assert payload["faces_detected"] >= 1
    for model in payload["models"].values():
        assert model["verdict"] in {"REAL", "FAKE"}
        assert 0.0 <= model["prob_fake"] <= 100.0
        assert model["prob_fake"] + model["prob_real"] == pytest.approx(100.0, abs=0.2)


def test_predict_is_deterministic(client, fake_sample):
    first = _upload(client, fake_sample.read_bytes()).get_json()
    second = _upload(client, fake_sample.read_bytes()).get_json()

    if first["models"]["cnn_simple"]["verdict"] == "—":
        pytest.skip("CNN indisponible")
    assert first["models"]["cnn_simple"]["prob_fake"] == second["models"]["cnn_simple"]["prob_fake"]
    assert first["models"]["resnet18"]["prob_fake"] == second["models"]["resnet18"]["prob_fake"]


def test_missing_file_is_rejected(client):
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_file"


def test_image_without_face_returns_422(client):
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (90, 90, 90)).save(buffer, format="PNG")
    response = _upload(client, buffer.getvalue(), "flat.png")

    assert response.status_code == 422
    assert response.get_json()["error"] == "no_face"


def test_unreadable_file_returns_400(client):
    response = _upload(client, b"definitely not an image", "broken.bin")
    assert response.status_code == 400
    assert response.get_json()["error"] == "unreadable"


def test_api_404_is_json(client):
    response = client.get("/api/unknown")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_payload_too_large_returns_413(client):
    original = application.app.config["MAX_CONTENT_LENGTH"]
    application.app.config["MAX_CONTENT_LENGTH"] = 1024
    try:
        response = _upload(client, b"x" * 4096, "big.jpg")
        assert response.status_code == 413
    finally:
        application.app.config["MAX_CONTENT_LENGTH"] = original


def test_demo_runs_on_a_repository_sample(client):
    response = client.post("/demo")
    assert response.status_code == 200
