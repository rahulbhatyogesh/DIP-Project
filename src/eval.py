"""Evaluation: accuracy and AUC, at both frame and video level.

The paper reports Accuracy and AUC (Tables II and III) without stating whether
the figures are per-frame or per-video, so both are computed.  Video-level
scores average the sigmoid outputs over a clip's 32 frames, which is the usual
protocol for Celeb-DF and is generally the stronger of the two.

Standalone use (scores a checkpoint on the test partition):

    python -m src.eval --ckpt runs/xception_hfc_identity/best.pt \
        --faces-root faces --manifest faces/manifest.csv --splits splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CelebDFFaces
from src.models.build import build_model


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, balanced accuracy and ROC-AUC from sigmoid scores.

    `balanced_accuracy` -- the mean of the two per-class recalls -- matters here
    because Celeb-DF-v2 is ~1:9.5 real:fake, so a test partition inherits that
    imbalance and plain accuracy is dominated by the fake class.  Always
    predicting "fake" scores ~0.917 on this data.

    The paper reports 0.8834 accuracy for XceptionNet on the identity split,
    which is BELOW that trivial baseline -- strong evidence its accuracies are
    computed on balanced data.  `balanced_accuracy` is therefore the figure to
    compare against the paper; `accuracy` is kept for completeness.
    """
    preds = (scores >= threshold).astype(np.int64)
    acc = float((preds == labels).mean())

    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")   # recall on fakes
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")   # recall on reals
    balanced_acc = float((tpr + tnr) / 2)

    # AUC via the rank identity (equivalent to sklearn, no dependency needed).
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        order = scores.argsort()
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        # Average ranks within ties so the result matches sklearn exactly.
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
        auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "recall_fake": float(tpr),
        "recall_real": float(tnr),
        "auc": auc,
        "n": int(len(labels)),
        "n_real": int(tn + fp),
        "n_fake": int(tp + fn),
        "majority_baseline": float(max(tp + fn, tn + fp) / len(labels)),
    }


def aggregate_by_video(
    scores: np.ndarray, labels: np.ndarray, video_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean sigmoid score per clip."""
    acc: dict[str, list[float]] = defaultdict(list)
    lab: dict[str, int] = {}
    for s, y, v in zip(scores, labels, video_ids):
        acc[v].append(float(s))
        lab[v] = int(y)
    vids = sorted(acc)
    return (
        np.array([np.mean(acc[v]) for v in vids]),
        np.array([lab[v] for v in vids], dtype=np.int64),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = True,
) -> dict:
    """Run the model over a loader and return frame- and video-level metrics."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_videos: list[str] = []
    total_loss, n_batches = 0.0, 0
    criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = amp and device.type == "cuda"

    for batch in loader:
        if len(batch) == 3:
            imgs, labels, vids = batch
            all_videos.extend(vids)
        else:
            imgs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(imgs).squeeze(1)
            loss = criterion(logits.float(), labels)

        total_loss += float(loss)
        n_batches += 1
        all_scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.append(labels.cpu().numpy().astype(np.int64))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    out = {
        "loss": total_loss / max(1, n_batches),
        "frame": binary_metrics(scores, labels),
    }
    if all_videos:
        v_scores, v_labels = aggregate_by_video(scores, labels, all_videos)
        out["video"] = binary_metrics(v_scores, v_labels)
    return out


def format_metrics(m: dict) -> str:
    parts = [f"loss={m['loss']:.4f}"]
    for level in ("frame", "video"):
        if level in m:
            d = m[level]
            parts.append(
                f"{level}: acc={d['accuracy']:.4f} bal_acc={d['balanced_accuracy']:.4f} "
                f"auc={d['auc']:.4f}"
            )
    return "  ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a checkpoint on a partition.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--faces-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="write metrics JSON here")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_model(cfg["backbone"], cfg["mode"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    ds = CelebDFFaces(
        args.faces_root, args.manifest, args.splits,
        cfg["protocol"], args.part, return_meta=True,
    )
    print(ds.describe())
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    metrics = evaluate(model, loader, device)
    metrics["run"] = f"{cfg['backbone']}/{cfg['mode']}/{cfg['protocol']}"
    metrics["part"] = args.part
    metrics["iteration"] = ckpt.get("iteration")
    print(f"{metrics['run']} [{args.part}]  {format_metrics(metrics)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
