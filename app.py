"""
=============================================================================
Application Web Flask : Détection de Deepfakes (Celeb-DF v2)
Comparaison en direct : CNN Simple (Étudiant 1) vs ResNet-18 (Étudiant 2)

Corrections apportées par rapport à la version précédente :
  · détection + recadrage du visage avant l'inférence (YuNet, OpenCV) ;
  · CNN chargé en mode déterministe (les couches d'augmentation sont retirées
    et un test de déterminisme est exécuté au démarrage) ;
  · chargement des modèles explicite : plus de repli silencieux sur « — » ;
  · carte de modèle partagée (model_card.json) : taille, normalisation, seuil ;
  · codes HTTP corrects (400 / 413 / 415 / 422 / 503) et endpoint /health ;
  · pipeline d'image partagé via le package `deepfake` (mêmes règles que le
    notebook) ;
  · gabarit HTML sorti de ce fichier (templates/index.html).
=============================================================================
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request
from PIL import Image

from deepfake.config import BASE_DIR, load_model_card
from deepfake.face import FaceDetector, FaceDetectorUnavailable, largest_face
from deepfake.models import check_determinism, load_cnn, load_resnet18
from deepfake.preprocess import (
    ImageDecodeError,
    cnn_batch,
    open_image,
    resnet_tensor,
    to_rgb,
)

# ---------------------------------------------------------------------------
# Journalisation
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s %(name)s : %(message)s",
)
log = logging.getLogger("deepfake.app")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CARD = load_model_card()
RESNET_MODEL_PATH = BASE_DIR / "resnet18.pt"
CNN_MODEL_PATH = BASE_DIR / "cnn_simple.keras"
SAMPLES_DIR = BASE_DIR / "echantillons_20_photos"
MAX_VIDEO_FRAMES = 5

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 Mo


# ---------------------------------------------------------------------------
# Erreurs métier
# ---------------------------------------------------------------------------
class NoFaceDetected(Exception):
    """Aucun visage exploitable dans l'image."""


class FaceDetectorMissing(Exception):
    """Le détecteur de visages n'est pas disponible."""


class VideoDecodeError(Exception):
    """Le fichier ressemble à une vidéo mais aucune frame n'a pu être lue."""


# ---------------------------------------------------------------------------
# État runtime (chargé une seule fois, thread-safe)
# ---------------------------------------------------------------------------
@dataclass
class RuntimeState:
    resnet: object | None = None
    cnn: object | None = None
    detector: FaceDetector | None = None
    errors: dict[str, str] = field(default_factory=dict)
    device: str = "cpu"
    torch_version: str = "N/A"
    tf_version: str = "N/A"
    cnn_stochastic_layers: list[str] = field(default_factory=list)


_state = RuntimeState()
_state_lock = threading.Lock()
_inference_lock = threading.Lock()  # sérialise l'inférence : modèles non réentrants
_loaded = False


def pick_device():
    """CUDA → MPS → CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_runtime() -> RuntimeState:
    """Charge modèles et détecteur une seule fois ; les erreurs sont exposées."""
    global _loaded
    with _state_lock:
        if _loaded:
            return _state
        _loaded = True

        try:
            import torch

            _state.torch_version = torch.__version__
            _state.device = str(pick_device())
        except ImportError as exc:
            _state.errors["torch"] = f"PyTorch indisponible : {exc}"
            log.error("PyTorch indisponible : %s", exc)

        if "torch" not in _state.errors:
            try:
                _state.resnet = load_resnet18(RESNET_MODEL_PATH, _state.device)
                log.info("ResNet-18 chargé sur %s", _state.device)
            except Exception as exc:  # checkpoint absent / incompatible
                _state.errors["resnet"] = f"{type(exc).__name__}: {exc}"
                log.error("Échec du chargement ResNet-18 : %s", exc, exc_info=True)

        try:
            import tensorflow as tf

            _state.tf_version = tf.__version__
        except Exception as exc:  # TF absent ou cassé : le CNN est facultatif
            _state.errors["cnn"] = f"TensorFlow indisponible : {exc}"
            log.warning("TensorFlow indisponible : %s", exc)
        else:
            try:
                model, removed = load_cnn(CNN_MODEL_PATH, deterministic=True)
                if not check_determinism(lambda batch: model.predict(batch, verbose=0)):
                    raise RuntimeError(
                        "le modèle reste non déterministe après retrait des couches aléatoires"
                    )
                _state.cnn = model
                _state.cnn_stochastic_layers = removed
                log.info("CNN Simple chargé (couches retirées : %s)", ", ".join(removed) or "aucune")
            except Exception as exc:
                _state.errors["cnn"] = f"{type(exc).__name__}: {exc}"
                log.error("Échec du chargement CNN : %s", exc, exc_info=True)

        try:
            detector = FaceDetector()
            detector.load()
            _state.detector = detector
            log.info("Détecteur de visages chargé")
        except FaceDetectorUnavailable as exc:
            _state.errors["face_detector"] = str(exc)
            log.error("Détecteur de visages indisponible : %s", exc)

        return _state


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
def format_prediction(prob: float, threshold: float) -> dict:
    """Convertit une probabilité « classe 1 » en verdict lisible."""
    probability = float(np.clip(prob, 0.0, 1.0))
    prob_fake = probability if CARD.class_1_is_fake else 1.0 - probability
    prob_real = 1.0 - prob_fake

    if prob_fake >= threshold:
        verdict, confidence = "FAKE", prob_fake
    else:
        verdict, confidence = "REAL", prob_real

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100.0, 1),
        "prob_fake": round(prob_fake * 100.0, 1),
        "prob_real": round(prob_real * 100.0, 1),
        "threshold": threshold,
        "latency_ms": 0.0,
    }


def _unavailable(reason: str) -> dict:
    return {
        "verdict": "—",
        "confidence": 0.0,
        "prob_fake": 0.0,
        "prob_real": 0.0,
        "threshold": CARD.threshold,
        "latency_ms": 0.0,
        "error": reason,
    }


def _predict_resnet(face: Image.Image) -> float:
    import torch

    state = load_runtime()
    if state.resnet is None:
        raise RuntimeError(state.errors.get("resnet", "ResNet-18 indisponible"))

    tensor = resnet_tensor(face, CARD.img_size, CARD.mean, CARD.std).to(state.device)
    with torch.inference_mode():
        logits = state.resnet(tensor)
    return float(torch.sigmoid(logits).squeeze().item())


def _predict_cnn(face: Image.Image) -> float:
    state = load_runtime()
    if state.cnn is None:
        raise RuntimeError(state.errors.get("cnn", "CNN indisponible"))

    batch = cnn_batch(face, CARD.img_size)
    output = state.cnn.predict(batch, verbose=0)
    return float(output.ravel()[0])


def _run_model(name: str, face: Image.Image) -> dict:
    """Inférence chronométrée d'un modèle, avec repli CPU explicite."""
    predictor = _predict_resnet if name == "resnet" else _predict_cnn
    state = load_runtime()

    try:
        started = time.perf_counter()
        probability = predictor(face)
        elapsed = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        log.error("Inférence %s en échec : %s", name, exc)
        if name == "resnet" and state.device != "cpu":
            # Un backend GPU peut échouer sur un tenseur donné : on repasse
            # durablement le modèle sur CPU au lieu de réessayer à chaque requête.
            previous_device = state.device
            try:
                state.resnet.to("cpu")
                state.device = "cpu"
                started = time.perf_counter()
                probability = _predict_resnet(face)
                elapsed = (time.perf_counter() - started) * 1000.0
                log.warning("ResNet-18 basculé sur CPU après échec du backend %s", previous_device)
            except Exception as fallback_exc:
                return _unavailable(f"{type(fallback_exc).__name__}: {fallback_exc}")
        else:
            return _unavailable(f"{type(exc).__name__}: {exc}")

    result = format_prediction(probability, CARD.threshold)
    result["latency_ms"] = round(elapsed, 1)
    return result


def encode_preview(face: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    face.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _consensus(first: dict, second: dict) -> bool | None:
    verdicts = {first["verdict"], second["verdict"]}
    if verdicts & {"—", "ERR"}:
        return None
    return first["verdict"] == second["verdict"]


def analyze_image(image: Image.Image, truth: str | None = None) -> dict:
    """Détecte le visage principal, recadre, puis interroge les deux modèles."""
    state = load_runtime()

    if state.detector is None:
        raise FaceDetectorMissing(
            state.errors.get(
                "face_detector",
                "Détecteur de visages indisponible (voir /health).",
            )
        )

    rgb = to_rgb(image)
    face, _, faces_detected = largest_face(rgb, state.detector, CARD.face_crop_margin)
    if face is None:
        raise NoFaceDetected(
            "Aucun visage détecté. Utilisez une photo où le visage est visible "
            "(l'application recadre automatiquement)."
        )

    with _inference_lock:
        resnet_result = _run_model("resnet", face)
        cnn_result = _run_model("cnn", face)

    return {
        "resnet": resnet_result,
        "cnn": cnn_result,
        "consensus": _consensus(resnet_result, cnn_result),
        "image_b64": encode_preview(face),
        "faces_detected": faces_detected,
        "crop_size": CARD.img_size,
        "threshold": CARD.threshold,
        "truth": truth,
        "frames": None,
    }


# ---------------------------------------------------------------------------
# Vidéo (facultative : OpenCV + frames échantillonnées)
# ---------------------------------------------------------------------------
def extract_video_frames(data: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[Image.Image]:
    """Échantillonne quelques frames réparties sur la durée de la vidéo."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise VideoDecodeError(
            "Support vidéo indisponible : pip install opencv-python-headless"
        ) from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(data)
            tmp_path = handle.name

        capture = cv2.VideoCapture(tmp_path)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                return []
            indices = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
            frames: list[Image.Image] = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return frames
        finally:
            capture.release()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def analyze_video(frames: list[Image.Image]) -> dict:
    """Moyenne les probabilités sur les frames où un visage est détecté."""
    state = load_runtime()
    if state.detector is None:
        raise FaceDetectorMissing(state.errors.get("face_detector", "Détecteur indisponible"))

    resnet_scores, cnn_scores, face_counts = [], [], []
    face: Image.Image | None = None

    for frame in frames:
        crop, _, count = largest_face(to_rgb(frame), state.detector, CARD.face_crop_margin)
        if crop is None:
            continue
        face_counts.append(count)
        face = face or crop
        with _inference_lock:
            if state.resnet is not None:
                try:
                    resnet_scores.append(_predict_resnet(crop))
                except Exception as exc:
                    log.warning("Frame ignorée (ResNet) : %s", exc)
            if state.cnn is not None:
                try:
                    cnn_scores.append(_predict_cnn(crop))
                except Exception as exc:
                    log.warning("Frame ignorée (CNN) : %s", exc)

    if face is None:
        raise NoFaceDetected(
            "Aucun visage détecté dans les frames analysées. Essayez une image fixe."
        )

    resnet_result = (
        format_prediction(float(np.mean(resnet_scores)), CARD.threshold)
        if resnet_scores
        else _unavailable("aucune frame exploitable")
    )
    cnn_result = (
        format_prediction(float(np.mean(cnn_scores)), CARD.threshold)
        if cnn_scores
        else _unavailable("aucune frame exploitable")
    )

    return {
        "resnet": resnet_result,
        "cnn": cnn_result,
        "consensus": _consensus(resnet_result, cnn_result),
        "image_b64": encode_preview(face),
        "faces_detected": max(face_counts),
        "crop_size": CARD.img_size,
        "threshold": CARD.threshold,
        "truth": None,
        "frames": len(face_counts),
    }


def analyze_bytes(data: bytes, truth: str | None = None) -> dict:
    """Image ou vidéo : décode, puis analyse."""
    try:
        image = open_image(data)
    except ImageDecodeError:
        frames = extract_video_frames(data)
        if not frames:
            raise VideoDecodeError(
                "Fichier illisible : ni image valide, ni vidéo décodable."
            )
        return analyze_video(frames)
    return analyze_image(image, truth=truth)


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
def render_page(result: dict | None = None, status: int = 200):
    state = load_runtime()
    return (
        render_template(
            "index.html",
            result=result,
            resnet_ready=state.resnet is not None,
            cnn_ready=state.cnn is not None,
            model_errors=state.errors,
            torch_version=state.torch_version,
            tf_version=state.tf_version,
            device=state.device,
            img_size=CARD.img_size,
            threshold=CARD.threshold,
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        ),
        status,
    )


def wants_json() -> bool:
    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_page()

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        flash("Sélectionnez un fichier image ou vidéo.", "error")
        return redirect("/")

    try:
        result = analyze_bytes(uploaded.read())
    except NoFaceDetected as exc:
        flash(str(exc), "error")
        return render_page(status=422)
    except (FaceDetectorMissing, VideoDecodeError) as exc:
        log.error("Analyse impossible : %s", exc)
        flash(f"Analyse impossible : {exc}", "error")
        return render_page(status=503 if isinstance(exc, FaceDetectorMissing) else 400)
    except Image.DecompressionBombError:
        flash("Image trop grande (limite 40 Mpx).", "error")
        return render_page(status=400)
    except Exception as exc:  # noqa: BLE001 - on informe l'utilisateur
        log.exception("Erreur inattendue pendant l'analyse")
        flash(f"Erreur : {exc}", "error")
        return render_page(status=500)

    return render_page(result)


@app.route("/demo", methods=["POST"])
def demo():
    """Analyse un échantillon du dépôt et affiche la vérité terrain."""
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not samples:
        flash("Aucun échantillon de démonstration dans le dépôt.", "error")
        return render_page(status=404)

    sample = random.choice(samples)
    truth = "FAKE" if "_FAKE_" in sample.name else "REAL"
    try:
        result = analyze_image(open_image(sample), truth=truth)
    except (NoFaceDetected, FaceDetectorMissing) as exc:
        flash(str(exc), "error")
        return render_page(status=422)
    except Exception as exc:  # noqa: BLE001
        log.exception("Erreur pendant la démo")
        flash(f"Erreur démo : {exc}", "error")
        return render_page(status=500)

    flash(f"Échantillon : {sample.name} — vérité terrain : {truth}.", "info")
    return render_page(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"status": "error", "error": "missing_file"}), 400

    try:
        result = analyze_bytes(uploaded.read())
    except NoFaceDetected as exc:
        return jsonify({"status": "error", "error": "no_face", "message": str(exc)}), 422
    except FaceDetectorMissing as exc:
        return jsonify({"status": "error", "error": "detector_unavailable", "message": str(exc)}), 503
    except VideoDecodeError as exc:
        return jsonify({"status": "error", "error": "unreadable", "message": str(exc)}), 400
    except Image.DecompressionBombError:
        return jsonify({"status": "error", "error": "image_too_large"}), 400
    except Exception as exc:  # noqa: BLE001 - pas de fuite de trace interne
        log.exception("Erreur API")
        return jsonify({"status": "error", "error": "internal_error", "message": str(exc)}), 500

    return jsonify(
        {
            "status": "success",
            "consensus": result["consensus"],
            "faces_detected": result["faces_detected"],
            "frames": result["frames"],
            "threshold": result["threshold"],
            "models": {"resnet18": result["resnet"], "cnn_simple": result["cnn"]},
        }
    )


@app.route("/health")
def health():
    state = load_runtime()
    models = {
        "resnet18": state.resnet is not None,
        "cnn_simple": state.cnn is not None,
        "face_detector": state.detector is not None,
    }
    return (
        jsonify(
            {
                "status": "ok" if all(models.values()) else "degraded",
                "models": models,
                "errors": state.errors,
                "device": state.device,
                "cnn_stochastic_layers_removed": state.cnn_stochastic_layers,
                "model_card": {
                    "img_size": CARD.img_size,
                    "threshold": CARD.threshold,
                    "mean": list(CARD.mean),
                    "std": list(CARD.std),
                },
                "versions": {"torch": state.torch_version, "tensorflow": state.tf_version},
            }
        ),
        200 if all(models.values()) else 503,
    )


# ---------------------------------------------------------------------------
# Gestion des erreurs HTTP
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def handle_too_large(_error):
    message = f"Fichier trop volumineux (maximum {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} Mo)."
    if wants_json():
        return jsonify({"status": "error", "error": "file_too_large", "message": message}), 413
    flash(message, "error")
    return render_page(status=413)


@app.errorhandler(404)
def handle_not_found(_error):
    if wants_json():
        return jsonify({"status": "error", "error": "not_found"}), 404
    return render_page(status=404)


@app.errorhandler(500)
def handle_server_error(error):
    log.exception("Erreur serveur non gérée : %s", error)
    if wants_json():
        return jsonify({"status": "error", "error": "internal_error"}), 500
    flash("Erreur interne du serveur.", "error")
    return render_page(status=500)


# ---------------------------------------------------------------------------
# Entrée
# ---------------------------------------------------------------------------
def get_free_port(preferred: int = 5001) -> int:
    """AirPlay occupe 5000 sur macOS : on cherche un port libre."""
    for port in [preferred, 5001, 8080, 8000, 5050, 5000]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def main() -> None:
    port = get_free_port(int(os.environ.get("PORT", 5001)))
    host = os.environ.get("HOST", "127.0.0.1")

    if app.secret_key == "dev-only-change-me":
        log.warning("FLASK_SECRET_KEY non défini : clé de développement utilisée.")

    log.info("Chargement des modèles…")
    state = load_runtime()
    if state.errors:
        log.warning("Chargement partiel : %s", state.errors)

    print(f"\n  DEEPFAKE LAB — http://localhost:{port}  (device {state.device})\n")
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)


if __name__ == "__main__":
    main()
