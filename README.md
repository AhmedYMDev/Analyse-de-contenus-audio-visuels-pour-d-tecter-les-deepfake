# Deepfake Detection — Celeb-DF v2

Binary deepfake classification benchmark comparing two independent architectures trained on the [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) dataset, with a live Flask web application for side-by-side model comparison.

---

## Results

| Model | Accuracy | AUC | Precision (Fake) | Recall (Fake) | F1 (Fake) | F1 (Real) |
|---|---|---|---|---|---|---|
| **ResNet-18** (Étudiant 2) | **93.79 %** | **98.89 %** | 90.75 % | **97.50 %** | **94.00 %** | **93.55 %** |
| CNN Simple (Étudiant 1) | 89.77 % | 96.23 % | **91.95 %** | 87.15 % | 89.48 % | 90.04 % |

> Both models were evaluated on a held-out test split stratified by video identity to prevent data leakage.

---

## Project Structure

```
DeepLearningProjetCode/
├── app.py                              # Flask web application (single-file)
├── requirements.txt                    # Python dependencies
│
├── deepfake-resnet18-etudiant1.ipynb   # Training notebook — Étudiant 1
├── deepfake-resnet18-etudiant2.ipynb   # Training notebook — Étudiant 2
│
├── resnet18.pt                         # ResNet-18 weights (PyTorch)
├── cnn_simple.keras                    # CNN Simple weights (Keras 3)
│
├── echantillons_20_photos/             # 21 labelled test images (REAL / FAKE)
│
├── result_etudiant1/                   # Training artefacts — CNN Simple
│   ├── cnn_simple.keras
│   ├── cnn_simple_confusion.png
│   ├── cnn_simple_courbes.png
│   └── cnn_simple_history.csv
│
└── result_etudiant2/                   # Training artefacts — ResNet-18
    ├── resnet18.pt
    ├── resnet18_confusion.png
    ├── resnet18_courbes.png
    ├── resnet18_history.csv
    ├── resnet18_erreurs.png
    └── comparaison_modeles.csv
```

---

## Models

### Étudiant 1 — CNN Simple (`cnn_simple.keras`)

Custom convolutional network trained from scratch with Keras 3 / TensorFlow.

| Component | Detail |
|---|---|
| Input | 128 × 128 × 3, `float32` |
| Preprocessing (built-in) | `Rescaling(1./255)` layer |
| Backbone | 4 × `Conv2D → BatchNorm → ReLU → MaxPool` blocks |
| Head | `GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)` |
| Optimiser | Adam |
| Loss | Binary Cross-Entropy |
| Augmentation | RandomFlip, RandomRotation, RandomBrightness, RandomContrast |

### Étudiant 2 — ResNet-18 (`resnet18.pt`)

ImageNet-pretrained ResNet-18 fine-tuned with PyTorch.

| Component | Detail |
|---|---|
| Input | 128 × 128 × 3, normalised with ImageNet μ/σ |
| Backbone | `torchvision.models.resnet18` (pretrained ImageNet) |
| Head | `Dropout(0.5) → Linear(512, 1)` |
| Loss | Binary Cross-Entropy with Logits |
| Inference | `torch.sigmoid(logit)` |
| Device | CUDA → Apple MPS → CPU (auto-detected) |

---

## Dataset

**Celeb-DF v2** — a large-scale challenging deepfake video dataset.

| Split | Videos | Faces (approx.) |
|---|---|---|
| Celeb-real | 590 | ~225 k |
| Celeb-synthesis | 5 639 | ~2.1 M |
| YouTube-real | 300 | ~134 k |

- Faces extracted at **128 × 128 px**.
- Split **stratified by video identity** to avoid subject leakage between train / val / test.

---

## Web Application

A single-file Flask app (`app.py`) that runs both models in parallel and displays a side-by-side forensic dashboard.

### Features

- **Drag & drop** upload — instant client-side preview before submit
- **Dual inference** — both models run on each upload; results rendered simultaneously
- **Latency measurement** — per-model inference time in milliseconds
- **Consensus banner** — highlights agreement or divergence between models
- **Demo mode** — one-click synthetic sample to validate the pipeline
- **REST API** — `/api/predict` endpoint for scripted evaluation
- **Dark Forest design** — monospace forensic UI, zero AI-slop aesthetics
- **Accessibility** — `prefers-reduced-motion`, `prefers-reduced-transparency`, `prefers-contrast: more` all handled

### Routes

| Method | Path | Description |
|---|---|---|
| `GET / POST` | `/` | Main interface — upload & analyse |
| `POST` | `/demo` | Run inference on a synthetic sample |
| `POST` | `/api/predict` | JSON REST endpoint |

### API Example

```bash
curl -X POST http://localhost:5001/api/predict \
  -F "file=@echantillons_20_photos/01_REAL_id35_0008_face_15135.jpg"
```

```json
{
  "status": "success",
  "consensus": true,
  "models": {
    "resnet18": {
      "verdict": "REAL",
      "confidence": 91.3,
      "prob_real": 91.3,
      "prob_fake": 8.7,
      "latency_ms": 7.2
    },
    "cnn_simple": {
      "verdict": "REAL",
      "confidence": 88.5,
      "prob_real": 88.5,
      "prob_fake": 11.5,
      "latency_ms": 14.6
    }
  }
}
```

---

## Setup

### Prerequisites

- Python 3.10 – 3.14
- macOS / Linux / Windows (WSL)

### Installation

```bash
# Enter the project directory
cd DeepLearningProjetCode

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

> **macOS note:** AirPlay Receiver reserves port 5000. The app auto-detects a free port starting at **5001**.

### Run

```bash
python app.py
```

Open your browser at **http://localhost:5001**

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | `deepfake-detector-secret-key-2024` | Flask session secret |
| `PORT` | `5001` | Preferred port (auto-fallback if occupied) |

---

## Image Preprocessing Pipeline

```
Upload (any resolution, any format)
  │
  ├─ EXIF orientation correction       (smartphone photos)
  ├─ Alpha compositing on white bg     (PNG / WEBP with transparency)
  ├─ RGB conversion
  └─ Resize → 128 × 128 px            (bilinear interpolation)
              │
              ├─ PyTorch tensor
              │   └─ Normalize(mean=[0.485,0.456,0.406],
              │                 std=[0.229,0.224,0.225])  → ResNet-18
              │
              └─ NumPy float32 [0–255]
                  └─ Internal Rescaling(1./255) layer     → CNN Simple
```

---

## Dependencies

```
flask>=3.0.0
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.15.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0   # optional — video frame extraction only
```

---

## Reproducing Training

Open the corresponding notebook and run all cells:

| Notebook | Model |
|---|---|
| `deepfake-resnet18-etudiant1.ipynb` | CNN Simple (Keras) |
| `deepfake-resnet18-etudiant2.ipynb` | ResNet-18 (PyTorch) |

The notebooks expect the Celeb-DF v2 dataset extracted locally (path configured in the first cell). The dataset root is excluded from version control via `.gitignore` (`Celeb_V2/`).

---

## Sample Images

The `echantillons_20_photos/` directory contains 21 pre-labelled face crops for quick manual testing.

Filename convention:

```
{index}_{REAL|FAKE}_{video_id}_face_{frame}.jpg

01_REAL_id35_0008_face_15135.jpg
02_FAKE_id33_id32_0005_face_26025.jpg
...
```

---

## License

This project is a university assignment. Model weights and code are for academic use only.  
The Celeb-DF v2 dataset is subject to its own [terms of use](https://github.com/yuezunli/celeb-deepfakeforensics).

