"""
=============================================================================
Application Web Flask : Détection de Deepfakes (Celeb-DF v2)
Comparaison en direct : CNN Simple (Étudiant 1) vs ResNet18 (Étudiant 2)
Design : Dark Forest / Forensic Minimal — Zéro AI-slop
=============================================================================
"""

import os
import io
import time
import base64
import socket
import threading
import numpy as np
from PIL import Image, ImageOps

from flask import Flask, request, render_template_string, flash, redirect, jsonify

# ---------------------------------------------------------------------------
# Imports Deep Learning (PyTorch & TensorFlow/Keras)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

import tensorflow as tf

# Import optionnel d'OpenCV pour le support vidéo
try:
    import cv2  # type: ignore
    HAS_OPENCV = True
except (ImportError, ModuleNotFoundError):
    cv2 = None
    HAS_OPENCV = False

# Verrou thread-safety
model_inference_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Configuration & Constantes
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "deepfake-detector-secret-key-2024")
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 Mo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet18.pt")
CNN_MODEL_PATH    = os.path.join(BASE_DIR, "cnn_simple.keras")

# Convention Celeb-DF v2 : classe 1 = FAKE
CLASS_1_IS_FAKE = True

# Périphérique PyTorch
if torch.cuda.is_available():
    PYTORCH_DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    PYTORCH_DEVICE = torch.device("mps")
else:
    PYTORCH_DEVICE = torch.device("cpu")

print(f"[*] Device PyTorch : {PYTORCH_DEVICE}")

# ---------------------------------------------------------------------------
# Chargement des modèles (une seule fois au démarrage)
# ---------------------------------------------------------------------------
resnet_model = None
cnn_model    = None


def load_models():
    global resnet_model, cnn_model

    if os.path.exists(RESNET_MODEL_PATH):
        try:
            print(f"[*] Chargement ResNet18 …")
            try:
                base_resnet = models.resnet18(weights=None)
            except TypeError:
                base_resnet = models.resnet18(pretrained=False)

            base_resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 1))

            checkpoint = torch.load(RESNET_MODEL_PATH, map_location=PYTORCH_DEVICE)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                base_resnet.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                base_resnet.load_state_dict(checkpoint, strict=False)
            else:
                base_resnet = checkpoint

            base_resnet.to(PYTORCH_DEVICE).eval()
            resnet_model = base_resnet
            print("[+] ResNet18 OK")
        except Exception as e:
            print(f"[!] ResNet18 erreur : {e}")

    if os.path.exists(CNN_MODEL_PATH):
        try:
            print("[*] Chargement CNN Simple …")
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            print("[+] CNN Simple OK")
        except Exception as e:
            print(f"[!] CNN Simple erreur : {e}")


load_models()

# ---------------------------------------------------------------------------
# Pipeline prétraitement
# ---------------------------------------------------------------------------
pytorch_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(pil_img):
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        alpha = pil_img.convert('RGBA')
        bg = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, alpha).convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    img_128 = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
    tensor  = pytorch_transforms(img_128).unsqueeze(0).to(PYTORCH_DEVICE)
    batch   = np.expand_dims(np.array(img_128, dtype=np.float32), axis=0)
    return tensor, batch, img_128


def extract_frame_from_video(file_bytes):
    if not HAS_OPENCV or cv2 is None:
        return None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        if ret and frame is not None:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[!] Vidéo : {e}")
    return None


def format_prediction(prob_val):
    p = max(0.0, min(1.0, float(prob_val)))
    p_fake = p if CLASS_1_IS_FAKE else 1.0 - p
    p_real = 1.0 - p_fake
    if p_fake >= 0.5:
        verdict, confidence = "FAKE", p_fake * 100.0
    else:
        verdict, confidence = "REAL", p_real * 100.0
    return {
        "verdict":    verdict,
        "confidence": round(confidence, 1),
        "prob_fake":  round(p_fake * 100.0, 1),
        "prob_real":  round(p_real * 100.0, 1),
    }


def execute_inference(pil_img):
    """Inférence thread-safe sur les deux modèles avec mesure de latence."""
    tensor, batch, img_128 = preprocess_image(pil_img)

    with model_inference_lock:
        # ── ResNet-18 ──────────────────────────────────────────────────────
        if resnet_model is not None:
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = resnet_model(tensor)
                    p   = torch.sigmoid(out).squeeze().item() if out.shape[-1] == 1 \
                          else torch.softmax(out, dim=1).squeeze()[1].item()
                resnet_ms     = round((time.perf_counter() - t0) * 1000, 1)
                resnet_result = format_prediction(p)
                resnet_result["latency_ms"] = resnet_ms
            except Exception as e:
                try:
                    dev = torch.device("cpu")
                    resnet_model.to(dev)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = resnet_model(tensor.to(dev))
                        p   = torch.sigmoid(out).squeeze().item() if out.shape[-1] == 1 \
                              else torch.softmax(out, dim=1).squeeze()[1].item()
                    resnet_ms     = round((time.perf_counter() - t0) * 1000, 1)
                    resnet_result = format_prediction(p)
                    resnet_result["latency_ms"] = resnet_ms
                except Exception:
                    resnet_result = {"verdict": "ERR", "confidence": 0.0,
                                     "prob_fake": 0.0, "prob_real": 0.0, "latency_ms": 0.0}
        else:
            resnet_result = {"verdict": "—", "confidence": 0.0,
                             "prob_fake": 0.0, "prob_real": 0.0, "latency_ms": 0.0}

        # ── CNN Simple ─────────────────────────────────────────────────────
        if cnn_model is not None:
            try:
                t0  = time.perf_counter()
                out = cnn_model.predict(batch, verbose=0)
                p   = float(out[0][0]) if out.shape[-1] == 1 else float(out[0][1])
                cnn_ms     = round((time.perf_counter() - t0) * 1000, 1)
                cnn_result = format_prediction(p)
                cnn_result["latency_ms"] = cnn_ms
            except Exception:
                cnn_result = {"verdict": "ERR", "confidence": 0.0,
                              "prob_fake": 0.0, "prob_real": 0.0, "latency_ms": 0.0}
        else:
            cnn_result = {"verdict": "—", "confidence": 0.0,
                          "prob_fake": 0.0, "prob_real": 0.0, "latency_ms": 0.0}

    buf = io.BytesIO()
    img_128.save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return resnet_result, cnn_result, img_b64


# ---------------------------------------------------------------------------
# Template HTML — Dark Forest / Forensic Minimal
# Design principles (Apple WWDC "Designing Fluid Interfaces"):
#   · Response on pointer-down, not release
#   · Spring easing (cubic-bezier .19,1,.22,1) for all transitions
#   · scaleX bar animation from transform-origin:left (direction hint)
#   · Symmetric enter/exit paths for preview swap
#   · Translucent header via backdrop-filter (material hierarchy)
#   · prefers-reduced-motion + prefers-reduced-transparency support
#   · Optical typography: tight tracking on large, open on small
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEEPFAKE LAB // FORENSIC BENCHMARK</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
/* ── Fonts ───────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  /* Dark Forest palette */
  --bg:        #080d0b;
  --surface:   #0f1a14;
  --card:      #111d17;
  --border:    #1a2e22;
  --border-hi: #234733;
  --fg:        #c8d8cc;
  --fg-muted:  #6b8f76;
  --fg-dim:    #3d5945;

  /* Accents */
  --green:     #34d399;
  --green-hi:  #10b981;
  --green-lo:  #064e3b;

  /* Status */
  --real-text: #34d399;
  --real-bg:   rgba(6,78,59,.55);
  --real-bdr:  #065f46;
  --fake-text: #f87171;
  --fake-bg:   rgba(69,10,10,.55);
  --fake-bdr:  #7f1d1d;

  /* Motion */
  --spring:    cubic-bezier(.19,1,.22,1);
  --snap:      cubic-bezier(.32,1,.22,1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html { background: var(--bg); color: var(--fg); }

body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  -webkit-font-smoothing: antialiased;
}

/* Mono helper */
.mono { font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace; }

/* ── Optical typography ──────────────────────────────────────────────── */
/* Large: tight tracking + leading (Apple §15) */
.text-display { letter-spacing: -.025em; line-height: 1.08; }
/* Body: neutral */
.text-body    { letter-spacing: 0;       line-height: 1.6; }
/* Caption: open tracking for legibility */
.text-caption { letter-spacing: .02em;   line-height: 1.45; }

/* ── Translucent header material (Apple §12) ─────────────────────────── */
.header-glass {
  background: rgba(8,13,11,.78);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border-bottom: 1px solid var(--border);
}
/* Scroll-edge fade instead of hard border */
.header-glass::after {
  content: '';
  position: absolute; bottom: -6px; left: 0; right: 0; height: 6px;
  background: linear-gradient(to bottom, rgba(8,13,11,.35), transparent);
  pointer-events: none;
}

/* ── Cards ───────────────────────────────────────────────────────────── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
}
.card-hover {
  transition: transform .35s var(--spring), border-color .35s ease, box-shadow .35s ease;
}
.card-hover:hover {
  transform: translateY(-2px);
  border-color: var(--border-hi);
  box-shadow: 0 12px 32px rgba(0,0,0,.35);
}

/* ── Progress bars ───────────────────────────────────────────────────── */
.bar-track {
  height: 3px;
  background: rgba(255,255,255,.05);
  border-radius: 99px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 99px;
  transform-origin: left;        /* direction hint (Apple §8) */
  transform: scaleX(0);
  transition: transform .9s var(--spring);
}
.bar-fill.ready { transform: scaleX(1); }

/* ── Verdict badges ──────────────────────────────────────────────────── */
.badge-real { background: var(--real-bg); color: var(--real-text); border: 1px solid var(--real-bdr); }
.badge-fake { background: var(--fake-bg); color: var(--fake-text); border: 1px solid var(--fake-bdr); }
.badge-warn { background: rgba(120,53,15,.4); color: #fbbf24; border: 1px solid #78350f; }

/* ── Buttons (press on pointer-down — Apple §1) ──────────────────────── */
.btn {
  cursor: pointer;
  border: none;
  transition: transform 110ms var(--snap), opacity 110ms ease;
}
.btn:active { transform: scale(.97); }

/* ── Drop zone ───────────────────────────────────────────────────────── */
.drop-zone {
  border: 1.5px dashed var(--border-hi);
  border-radius: 10px;
  transition: border-color .2s ease, background .2s ease;
}
.drop-zone.over {
  border-color: var(--green-hi);
  background: rgba(16,185,129,.04);
}

/* ── Page animations ─────────────────────────────────────────────────── */
@keyframes fade-up {
  from { opacity: 0; transform: translateY(10px) scale(.99); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes materialize {
  from { opacity: 0; transform: scale(.98); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes scan-line {
  0%   { transform: translateY(-100%); opacity: .0; }
  10%  { opacity: .15; }
  90%  { opacity: .15; }
  100% { transform: translateY(600%); opacity: .0; }
}

.anim-up          { animation: fade-up    .45s var(--spring) both; }
.anim-up-d1       { animation: fade-up    .45s var(--spring) .07s both; }
.anim-up-d2       { animation: fade-up    .45s var(--spring) .14s both; }
.anim-materialize { animation: materialize .4s var(--spring) both; }

/* Scan-line overlay on result image */
.img-wrap { position: relative; overflow: hidden; }
.img-wrap::after {
  content: '';
  position: absolute; left: 0; right: 0; height: 2px;
  background: linear-gradient(to right, transparent, var(--green-hi), transparent);
  animation: scan-line 2s ease-in-out 1 .3s;
  pointer-events: none;
}

/* ── Status dot ──────────────────────────────────────────────────────── */
@keyframes breathe {
  0%,100% { opacity: .35; }
  50%      { opacity: 1; }
}
.dot-live { animation: breathe 2.8s ease-in-out infinite; }

/* ── Reduced motion (Apple §14) ─────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 150ms !important;
  }
  .bar-fill { transition: none; }
  .card-hover:hover { transform: none; }
  .btn:active { transform: none; }
}

/* ── Reduced transparency (Apple §14) ───────────────────────────────── */
@media (prefers-reduced-transparency: reduce) {
  .header-glass { background: var(--bg) !important; backdrop-filter: none !important; }
}

/* ── High contrast (Apple §14) ──────────────────────────────────────── */
@media (prefers-contrast: more) {
  :root { --border: rgba(255,255,255,.25); --fg-muted: #9ab8a2; }
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 99px; }
</style>
</head>

<body>
<!-- ═══════════════════════════════════════════════════════════════════
     HEADER — 100% typographic, no icons, no logos (Forensic Minimal)
═══════════════════════════════════════════════════════════════════ -->
<header class="header-glass sticky top-0 z-50 relative">
  <div style="max-width:1100px;margin:0 auto;padding:0 1.5rem;height:48px;display:flex;align-items:center;justify-content:space-between;">

    <div style="display:flex;align-items:baseline;gap:.75rem;">
      <span class="mono" style="font-size:13px;font-weight:700;color:var(--green);letter-spacing:.06em;">DEEPFAKE LAB</span>
      <span class="mono" style="font-size:11px;color:var(--fg-dim);letter-spacing:.04em;">//</span>
      <span class="mono" style="font-size:11px;color:var(--fg-muted);letter-spacing:.04em;">FORENSIC BENCHMARK</span>
    </div>

    <div style="display:flex;align-items:center;gap:.5rem;">
      <span class="dot-live" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{% if resnet_ready and cnn_ready %}var(--green){% else %}#fbbf24{% endif %};"></span>
      <span class="mono text-caption" style="font-size:10px;color:var(--fg-muted);letter-spacing:.06em;">
        {% if resnet_ready and cnn_ready %}2 / 2 MODELS LOADED{% else %}PARTIAL LOAD{% endif %}
      </span>
    </div>

  </div>
</header>

<!-- ═══════════════════════════════════════════════════════════════════
     FLASH MESSAGES
═══════════════════════════════════════════════════════════════════ -->
{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    <div style="max-width:1100px;margin:1.25rem auto;padding:0 1.5rem;">
      {% for cat, msg in messages %}
        <div style="padding:.75rem 1rem;border-radius:8px;font-size:13px;border:1px solid;
                    {% if cat == 'error' %}background:var(--fake-bg);color:var(--fake-text);border-color:var(--fake-bdr);
                    {% else %}background:var(--real-bg);color:var(--real-text);border-color:var(--real-bdr);{% endif %}">
          {{ msg }}
        </div>
      {% endfor %}
    </div>
  {% endif %}
{% endwith %}

<!-- ═══════════════════════════════════════════════════════════════════
     MAIN
═══════════════════════════════════════════════════════════════════ -->
<main style="flex:1;max-width:1100px;margin:0 auto;width:100%;padding:2rem 1.5rem 3rem;">

  <!-- PAGE TITLE -->
  <div class="anim-up" style="margin-bottom:2.5rem;">
    <p class="mono text-caption" style="font-size:11px;color:var(--fg-dim);letter-spacing:.08em;margin-bottom:.5rem;">CELEB-DF V2 // BINARY CLASSIFICATION</p>
    <h1 class="text-display" style="font-size:clamp(1.6rem,3vw,2.25rem);font-weight:700;color:#e8f5ed;line-height:1.1;">
      Deepfake Detection<br>
      <span style="color:var(--green);font-weight:300;">Dual-Model Analysis</span>
    </h1>
  </div>

  <!-- GRID -->
  <div style="display:grid;grid-template-columns:1fr 1.5fr;gap:1.5rem;align-items:start;" id="mainGrid">

    <!-- ── LEFT: UPLOAD PANEL ─────────────────────────────────────── -->
    <div class="anim-up-d1">

      <!-- Drop Zone -->
      <div class="drop-zone" id="dropZone" style="padding:1.75rem 1.25rem;text-align:center;cursor:pointer;position:relative;margin-bottom:1rem;">
        <input type="file" id="fileInput" name="file" accept="image/*,video/mp4,video/quicktime"
               style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer;z-index:10;">

        <div id="dropPrompt">
          <!-- No icon — text-only prompt -->
          <p class="mono text-caption" style="font-size:11px;color:var(--fg-dim);letter-spacing:.05em;margin-bottom:.5rem;">DRAG &amp; DROP</p>
          <p style="font-size:14px;font-weight:500;color:var(--fg-muted);">or browse an image</p>
          <p class="mono text-caption" style="font-size:10px;color:var(--fg-dim);margin-top:.5rem;">JPG · PNG · WEBP · MP4</p>
        </div>

        <div id="previewWrap" style="display:none;flex-direction:column;align-items:center;gap:.75rem;">
          <div class="img-wrap" style="width:96px;height:96px;border-radius:8px;overflow:hidden;border:1px solid var(--border-hi);">
            <img id="clientPreview" style="width:100%;height:100%;object-fit:cover;" alt="">
          </div>
          <span id="fileName" class="mono text-caption" style="font-size:10px;color:var(--green);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
        </div>
      </div>

      <!-- Run button -->
      <form action="/" method="POST" enctype="multipart/form-data" id="uploadForm">
        <input type="file" name="file" id="hiddenFile" style="display:none;">
        <button type="submit" id="submitBtn" class="btn"
                style="width:100%;padding:.8rem;border-radius:8px;background:var(--green-hi);color:#fff;font-weight:600;font-size:14px;letter-spacing:.01em;">
          Run Analysis
        </button>
      </form>

      <!-- Demo -->
      <form action="/demo" method="POST" style="margin-top:.6rem;">
        <button type="submit" class="btn"
                style="width:100%;padding:.65rem;border-radius:8px;background:transparent;color:var(--fg-muted);font-size:12px;border:1px solid var(--border);letter-spacing:.01em;">
          Run demo sample
        </button>
      </form>

      <!-- Model status -->
      <div style="margin-top:1.75rem;padding-top:1.25rem;border-top:1px solid var(--border);">
        <p class="mono text-caption" style="font-size:10px;color:var(--fg-dim);letter-spacing:.08em;margin-bottom:.75rem;">LOADED MODELS</p>

        {% for name, ready, meta in [
            ('CNN Baseline', cnn_ready, 'Keras · 4× Conv2D · Adam'),
            ('ResNet-18',    resnet_ready, 'PyTorch · ImageNet · Transfer')
           ] %}
        <div style="display:flex;align-items:center;justify-content:space-between;padding:.55rem .7rem;border-radius:6px;border:1px solid var(--border);background:var(--surface);margin-bottom:.5rem;">
          <div style="display:flex;align-items:center;gap:.6rem;">
            <span style="width:5px;height:5px;border-radius:50%;background:{% if ready %}var(--green){% else %}var(--fake-text){% endif %};flex-shrink:0;"></span>
            <span style="font-size:12px;font-weight:500;color:{% if ready %}var(--fg){% else %}var(--fg-dim){% endif %};">{{ name }}</span>
          </div>
          <span class="mono text-caption" style="font-size:10px;color:var(--fg-dim);">{{ meta }}</span>
        </div>
        {% endfor %}
      </div>

    </div>

    <!-- ── RIGHT: RESULTS ────────────────────────────────────────── -->
    <div class="anim-up-d2">
      {% if result %}

        <!-- Consensus banner -->
        {% set consensus = (result.resnet.verdict == result.cnn.verdict) %}
        <div class="anim-materialize" style="margin-bottom:1.25rem;padding:1rem 1.25rem;border-radius:10px;
             border:1px solid {% if consensus and result.resnet.verdict == 'REAL' %}var(--real-bdr){% elif consensus and result.resnet.verdict == 'FAKE' %}var(--fake-bdr){% else %}var(--border-hi){% endif %};
             background:{% if consensus and result.resnet.verdict == 'REAL' %}var(--real-bg){% elif consensus and result.resnet.verdict == 'FAKE' %}var(--fake-bg){% else %}rgba(120,53,15,.3){% endif %};">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
            <div>
              <p class="mono text-caption" style="font-size:10px;letter-spacing:.08em;color:var(--fg-dim);margin-bottom:.3rem;">CONSENSUS</p>
              <p class="text-display" style="font-size:1.35rem;font-weight:700;
                 color:{% if consensus and result.resnet.verdict == 'REAL' %}var(--real-text){% elif consensus and result.resnet.verdict == 'FAKE' %}var(--fake-text){% else %}#fbbf24{% endif %};">
                {% if consensus %}
                  {{ 'AUTHENTIC' if result.resnet.verdict == 'REAL' else 'DEEPFAKE DETECTED' }}
                {% else %}
                  MODELS DIVERGE
                {% endif %}
              </p>
              <p style="font-size:12px;color:var(--fg-muted);margin-top:.3rem;">
                {% if consensus %}Both models agree on this sample.
                {% else %}ResNet-18 → <strong style="color:var(--fg);">{{ result.resnet.verdict }}</strong> &nbsp;·&nbsp; CNN → <strong style="color:var(--fg);">{{ result.cnn.verdict }}</strong>{% endif %}
              </p>
            </div>
            <div class="img-wrap" style="width:72px;height:72px;border-radius:8px;overflow:hidden;border:1px solid var(--border-hi);flex-shrink:0;">
              <img src="data:image/jpeg;base64,{{ result.image_b64 }}" style="width:100%;height:100%;object-fit:cover;" alt="analysed">
            </div>
          </div>
        </div>

        <!-- Model cards -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">

          {% for model_key, label, stack, meta_lines in [
              ('resnet', 'ResNet-18', 'Étudiant 2 — Transfer Learning',
               [('Backbone', 'ResNet-18'), ('Weights', 'ImageNet'), ('Runtime', 'PyTorch')]),
              ('cnn',    'CNN Baseline', 'Étudiant 1 — From Scratch',
               [('Architecture', '4× Conv2D'), ('Regularisation', 'Dropout'), ('Optimiser', 'Adam')])
             ] %}
          {% set m = result[model_key] %}
          <div class="card card-hover anim-materialize" style="padding:1.1rem 1rem;">

            <!-- Header -->
            <p class="mono text-caption" style="font-size:10px;letter-spacing:.07em;color:var(--fg-dim);margin-bottom:.25rem;">
              {{ stack.upper() }}
            </p>
            <p style="font-size:14px;font-weight:600;color:var(--fg);margin-bottom:.85rem;">{{ label }}</p>

            <!-- Verdict badge -->
            <div style="margin-bottom:1rem;">
              {% if m.verdict == 'REAL' %}
                <span class="badge-real" style="display:inline-block;padding:.35rem .9rem;border-radius:6px;font-weight:700;font-size:1.05rem;letter-spacing:.08em;" class="text-display">REAL</span>
              {% elif m.verdict == 'FAKE' %}
                <span class="badge-fake" style="display:inline-block;padding:.35rem .9rem;border-radius:6px;font-weight:700;font-size:1.05rem;letter-spacing:.08em;" class="text-display">FAKE</span>
              {% else %}
                <span class="badge-warn" style="display:inline-block;padding:.35rem .9rem;border-radius:6px;font-weight:600;font-size:.9rem;">{{ m.verdict }}</span>
              {% endif %}
            </div>

            <!-- Confidence -->
            <p class="mono" style="font-size:1.6rem;font-weight:700;color:{% if m.verdict == 'REAL' %}var(--real-text){% elif m.verdict == 'FAKE' %}var(--fake-text){% else %}#fbbf24{% endif %};letter-spacing:-.02em;line-height:1;margin-bottom:.2rem;">
              {{ m.confidence }}<span style="font-size:.85rem;font-weight:400;color:var(--fg-muted);margin-left:1px;">%</span>
            </p>
            <p class="text-caption" style="font-size:11px;color:var(--fg-dim);margin-bottom:1rem;">confidence</p>

            <!-- Probability bars -->
            <div style="margin-bottom:1rem;">
              <!-- FAKE bar -->
              <div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">
                <span class="text-caption" style="font-size:10px;color:var(--fg-dim);">FAKE</span>
                <span class="mono text-caption" style="font-size:10px;color:var(--fake-text);">{{ m.prob_fake }}%</span>
              </div>
              <div class="bar-track" style="margin-bottom:.6rem;">
                <div class="bar-fill" data-width="{{ m.prob_fake }}"
                     style="background:linear-gradient(to right,#7f1d1d,var(--fake-text));"></div>
              </div>
              <!-- REAL bar -->
              <div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">
                <span class="text-caption" style="font-size:10px;color:var(--fg-dim);">REAL</span>
                <span class="mono text-caption" style="font-size:10px;color:var(--real-text);">{{ m.prob_real }}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" data-width="{{ m.prob_real }}"
                     style="background:linear-gradient(to right,var(--green-lo),var(--green));"></div>
              </div>
            </div>

            <!-- Metadata table -->
            <div style="padding-top:.85rem;border-top:1px solid var(--border);">
              <p class="mono text-caption" style="font-size:9px;letter-spacing:.07em;color:var(--fg-dim);margin-bottom:.5rem;">METADATA</p>
              {% for k, v in meta_lines %}
              <div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">
                <span class="text-caption" style="font-size:10px;color:var(--fg-dim);">{{ k }}</span>
                <span class="mono text-caption" style="font-size:10px;color:var(--fg-muted);">{{ v }}</span>
              </div>
              {% endfor %}
              <div style="display:flex;justify-content:space-between;margin-top:.4rem;padding-top:.4rem;border-top:1px solid var(--border);">
                <span class="text-caption" style="font-size:10px;color:var(--fg-dim);">Latency</span>
                <span class="mono text-caption" style="font-size:10px;color:var(--green);">{{ m.latency_ms }} ms</span>
              </div>
            </div>

          </div>
          {% endfor %}

        </div>

      {% else %}

        <!-- Empty state — clear wayfinding (Apple §16) -->
        <div style="border:1.5px dashed var(--border);border-radius:12px;padding:3.5rem 2rem;text-align:center;min-height:380px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;">
          <p class="mono text-caption" style="font-size:11px;letter-spacing:.08em;color:var(--fg-dim);">AWAITING INPUT</p>
          <p style="font-size:14px;color:var(--fg-muted);max-width:280px;line-height:1.5;">
            Upload an image on the left or run the demo sample to compare both models.
          </p>
        </div>

      {% endif %}
    </div>

  </div>
</main>

<!-- ═══════════════════════════════════════════════════════════════════
     FOOTER
═══════════════════════════════════════════════════════════════════ -->
<footer style="border-top:1px solid var(--border);padding:.9rem 1.5rem;">
  <div style="max-width:1100px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
    <span class="mono text-caption" style="font-size:10px;color:var(--fg-dim);letter-spacing:.05em;">
      CELEB-DF V2 &nbsp;·&nbsp; BINARY CLASSIFICATION &nbsp;·&nbsp; 128×128 INPUT
    </span>
    <span class="mono text-caption" style="font-size:10px;color:var(--fg-dim);letter-spacing:.04em;">
      PyTorch&nbsp;{{ torch_version }}&nbsp;&nbsp;·&nbsp;&nbsp;TensorFlow&nbsp;{{ tf_version }}
    </span>
  </div>
</footer>

<!-- ═══════════════════════════════════════════════════════════════════
     JAVASCRIPT — Fluid interactions
     Principles applied:
       · Pointer-down response (Apple §1)
       · Symmetric enter/exit for preview swap (Apple §7)
       · scaleX bar animation triggered after paint (Apple §8)
       · Native drag & drop with instant highlight (Apple §2)
═══════════════════════════════════════════════════════════════════ -->
<script>
(function () {
  'use strict';

  /* ── Elements ── */
  const fileInput     = document.getElementById('fileInput');
  const hiddenFile    = document.getElementById('hiddenFile');
  const uploadForm    = document.getElementById('uploadForm');
  const submitBtn     = document.getElementById('submitBtn');
  const dropZone      = document.getElementById('dropZone');
  const dropPrompt    = document.getElementById('dropPrompt');
  const previewWrap   = document.getElementById('previewWrap');
  const clientPreview = document.getElementById('clientPreview');
  const fileName      = document.getElementById('fileName');

  /* ── Transfer files from overlay input to form input ── */
  function applyFile(file) {
    if (!file) return;
    fileName.textContent = file.name;

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        clientPreview.src = ev.target.result;
        /* Symmetric exit — fade-up out (Apple §7) */
        dropPrompt.style.transition = 'opacity 180ms ease, transform 180ms cubic-bezier(.19,1,.22,1)';
        dropPrompt.style.opacity    = '0';
        dropPrompt.style.transform  = 'translateY(-6px)';
        setTimeout(() => {
          dropPrompt.style.display   = 'none';
          previewWrap.style.display  = 'flex';
          /* Symmetric enter — fade-up in from same axis */
          previewWrap.style.opacity   = '0';
          previewWrap.style.transform = 'translateY(6px) scale(.97)';
          requestAnimationFrame(() => {
            previewWrap.style.transition  = 'opacity 300ms ease, transform 300ms cubic-bezier(.19,1,.22,1)';
            previewWrap.style.opacity     = '1';
            previewWrap.style.transform   = 'translateY(0) scale(1)';
          });
        }, 170);
      };
      reader.readAsDataURL(file);
    }

    /* Forward file to the form's hidden input via DataTransfer */
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      hiddenFile.files = dt.files;
    } catch (_) {}
  }

  /* ── File input change ── */
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      applyFile(e.target.files[0]);
    });
  }

  /* ── Submit: sync file & show loading state ── */
  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      /* Copy overlay input into hidden input if needed */
      if (fileInput && fileInput.files.length && !hiddenFile.files.length) {
        try {
          const dt = new DataTransfer();
          dt.items.add(fileInput.files[0]);
          hiddenFile.files = dt.files;
        } catch (_) {}
      }
      /* Loading feedback — pointer-down feel (Apple §1) */
      if (submitBtn) {
        submitBtn.disabled    = true;
        submitBtn.textContent = 'Running…';
        submitBtn.style.opacity = '.55';
      }
    });
  }

  /* ── Drag & drop: instant highlight (Apple §1) ── */
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(evt => {
      dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('over'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
      dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('over'); });
    });
    dropZone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      applyFile(file);
    });
  }

  /* ── Progress bars: scaleX from left after paint (Apple §8) ── */
  /* Bars start at scaleX(0) via CSS; trigger after layout is painted */
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.bar-fill').forEach(bar => {
        const w = parseFloat(bar.dataset.width || '0') / 100;
        bar.style.transform = `scaleX(${w})`;
      });
    });
  });

  /* ── Responsive grid collapse ── */
  function updateGrid() {
    const grid = document.getElementById('mainGrid');
    if (!grid) return;
    grid.style.gridTemplateColumns = window.innerWidth < 760 ? '1fr' : '1fr 1.5fr';
  }
  updateGrid();
  window.addEventListener('resize', updateGrid);

})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash("Aucun fichier reçu.", "error")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("Sélectionnez un fichier valide.", "error")
            return redirect(request.url)

        try:
            file_bytes = file.read()
            pil_image  = None
            try:
                pil_image = Image.open(io.BytesIO(file_bytes))
            except Exception:
                pil_image = extract_frame_from_video(file_bytes)

            if pil_image is None:
                flash("Impossible de lire ce fichier.", "error")
                return redirect(request.url)

            resnet_result, cnn_result, img_b64 = execute_inference(pil_image)
            result = {"resnet": resnet_result, "cnn": cnn_result, "image_b64": img_b64}

        except Exception as e:
            flash(f"Erreur : {e}", "error")
            return redirect(request.url)

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        resnet_ready=(resnet_model is not None),
        cnn_ready=(cnn_model is not None),
        torch_version=getattr(torch, '__version__', 'N/A'),
        tf_version=getattr(tf, '__version__', 'N/A'),
    )


@app.route("/demo", methods=["POST"])
def demo():
    try:
        img = Image.new('RGB', (128, 128), color=(180, 160, 140))
        resnet_result, cnn_result, img_b64 = execute_inference(img)
        result = {"resnet": resnet_result, "cnn": cnn_result, "image_b64": img_b64}
    except Exception as e:
        flash(f"Erreur démo : {e}", "error")
        result = None

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        resnet_ready=(resnet_model is not None),
        cnn_ready=(cnn_model is not None),
        torch_version=getattr(torch, '__version__', 'N/A'),
        tf_version=getattr(tf, '__version__', 'N/A'),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' key"}), 400
    try:
        pil_image = Image.open(io.BytesIO(request.files['file'].read()))
        resnet_r, cnn_r, _ = execute_inference(pil_image)
        return jsonify({
            "status": "success",
            "models": {"resnet18": resnet_r, "cnn_simple": cnn_r},
            "consensus": resnet_r["verdict"] == cnn_r["verdict"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point — auto-detect free port (macOS port 5000 = AirPlay)
# ---------------------------------------------------------------------------
def get_free_port(preferred=5001):
    for p in [preferred, 5001, 8080, 8000, 5050]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return preferred


if __name__ == "__main__":
    port = get_free_port(int(os.environ.get("PORT", 5001)))
    print(f"\n  DEEPFAKE LAB — http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
