#!/usr/bin/env python3
"""Réévalue les deux modèles sur le split de test, de façon reproductible.

Contrairement aux chiffres archivés, ce script :

* charge le CNN en mode déterministe (couches aléatoires retirées) ;
* calcule le seuil de décision sur la **validation**, puis l'applique au test ;
* rapporte les métriques au niveau **frame** *et* au niveau **vidéo** ;
* fournit un intervalle de confiance bootstrap et un test de McNemar.

    python scripts/evaluate_on_split.py \
        --dataset Celeb_V2 \
        --splits result_etudiant1/split_par_video.csv \
        --output result_etudiant2/reevaluation

Le dataset n'est pas inclus dans le dépôt : voir la section « Dataset » du README.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepfake import metrics  # noqa: E402
from deepfake.config import load_model_card  # noqa: E402
from deepfake.models import load_cnn, load_resnet18  # noqa: E402
from deepfake.preprocess import cnn_batch, open_image, prepare, resnet_tensor  # noqa: E402

BATCH_SIZE = 64


def read_split(csv_path: Path, split: str, dataset: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            row["path"] = dataset / row["rel_path"]
            row["label"] = int(row["label"])
            rows.append(row)
    if limit:
        rows = rows[:limit]
    return rows


def predict_all(rows: list[dict], card, device: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Probabilités des deux modèles, par lots."""
    import torch

    resnet = load_resnet18(ROOT / "resnet18.pt", device)
    cnn, removed = load_cnn(ROOT / "cnn_simple.keras", deterministic=True)
    print(f"[{tag}] CNN déterministe (couches retirées : {', '.join(removed) or 'aucune'})")

    probs_resnet, probs_cnn = [], []
    started = time.perf_counter()

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        faces = [prepare(open_image(row["path"]), card.img_size) for row in batch]

        tensors = torch.cat(
            [resnet_tensor(face, card.img_size, card.mean, card.std) for face in faces], dim=0
        ).to(device)
        with torch.inference_mode():
            logits = resnet(tensors)
        probs_resnet.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())

        batch_cnn = np.concatenate([cnn_batch(face, card.img_size) for face in faces], axis=0)
        probs_cnn.extend(cnn.predict(batch_cnn, verbose=0).ravel().tolist())

        done = min(start + BATCH_SIZE, len(rows))
        if (start // BATCH_SIZE) % 20 == 0 or done == len(rows):
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0
            print(f"[{tag}] {done}/{len(rows)} images ({rate:.0f} img/s)", flush=True)

    return np.asarray(probs_resnet), np.asarray(probs_cnn)


def evaluate(rows: list[dict], probs: np.ndarray, threshold: float) -> dict:
    y_true = np.array([row["label"] for row in rows])
    groups = [row["video"] for row in rows]

    frame = metrics.classification_metrics(y_true, probs, threshold)
    video_metrics, _ = metrics.evaluate_video_level(groups, y_true, probs, threshold)

    auc_low, auc_high = metrics.bootstrap_ci(metrics.roc_auc, y_true, probs, n_boot=500)
    return {
        "frame": frame,
        "video": video_metrics,
        "auc_ci95": [auc_low, auc_high],
        "ece": metrics.expected_calibration_error(y_true, probs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="racine contenant les images")
    parser.add_argument("--splits", type=Path, required=True, help="CSV rel_path,label,video,split")
    parser.add_argument("--output", type=Path, default=ROOT / "result_etudiant2" / "reevaluation")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (détection auto par défaut)")
    parser.add_argument("--threshold", type=float, default=None, help="seuil fixe ; sinon calculé sur val")
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que N images (test rapide)")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Dataset introuvable : {args.dataset}", file=sys.stderr)
        return 1
    if not args.splits.is_file():
        print(f"CSV de split introuvable : {args.splits}", file=sys.stderr)
        return 1

    card = load_model_card()

    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")

    test_rows = read_split(args.splits, "test", args.dataset, args.limit)
    if not test_rows:
        print("Aucune image de test trouvée dans le CSV.", file=sys.stderr)
        return 1
    print(f"Test : {len(test_rows)} images, {len({r['video'] for r in test_rows})} vidéos")

    # 1) Seuil sur la validation (jamais sur le test).
    threshold = args.threshold
    if threshold is None:
        val_rows = read_split(args.splits, "val", args.dataset, args.limit)
        if not val_rows:
            print("Validation absente : seuil par défaut 0.5", file=sys.stderr)
            threshold = card.threshold
        else:
            print(f"Validation : {len(val_rows)} images — calcul du seuil Youden…")
            val_probs, _ = predict_all(val_rows, card, device, "val")
            y_val = np.array([row["label"] for row in val_rows])
            threshold = metrics.best_threshold(y_val, val_probs, "youden")
            print(f"Seuil retenu (validation) : {threshold:.4f}")
    print(f"Seuil appliqué au test : {threshold:.4f}")

    # 2) Test.
    probs_resnet, probs_cnn = predict_all(test_rows, card, device, "test")
    y_true = np.array([row["label"] for row in test_rows])

    results = {
        "resnet18": evaluate(test_rows, probs_resnet, threshold),
        "cnn_simple": evaluate(test_rows, probs_cnn, threshold),
    }
    pred_resnet = (probs_resnet >= threshold).astype(int)
    pred_cnn = (probs_cnn >= threshold).astype(int)
    results["mcnemar_resnet_vs_cnn"] = metrics.mcnemar_test(y_true, pred_resnet, pred_cnn)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps({"threshold": threshold, "n_test": len(test_rows), **results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rel_path", "video", "label", "prob_resnet", "prob_cnn"])
        for row, prob_r, prob_c in zip(test_rows, probs_resnet, probs_cnn):
            writer.writerow([row["rel_path"], row["video"], row["label"], f"{prob_r:.6f}", f"{prob_c:.6f}"])

    # 3) Rapport lisible.
    print("\n=== Résultats (test, seuil {:.4f}) ===".format(threshold))
    header = f"{'Modèle':<12}{'Niveau':<8}{'Acc':>8}{'AUC':>8}{'P(FAKE)':>9}{'R(FAKE)':>9}{'F1(FAKE)':>10}"
    print(header)
    print("-" * len(header))
    for name, payload in results.items():
        if name.startswith("mcnemar"):
            continue
        for level in ("frame", "video"):
            m = payload[level]
            print(
                f"{name:<12}{level:<8}{m['accuracy']:>8.4f}{m['auc']:>8.4f}"
                f"{m['precision_fake']:>9.4f}{m['recall_fake']:>9.4f}{m['f1_fake']:>10.4f}"
            )
    mcnemar = results["mcnemar_resnet_vs_cnn"]
    print(
        f"\nMcNemar ResNet vs CNN : p = {mcnemar['p_value']:.4g} "
        f"(ResNet seul correct : {mcnemar['a_only']}, CNN seul correct : {mcnemar['b_only']})"
    )
    print(f"AUC ResNet IC95 : [{results['resnet18']['auc_ci95'][0]:.4f}, {results['resnet18']['auc_ci95'][1]:.4f}]")
    print(f"Artefacts écrits dans {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
