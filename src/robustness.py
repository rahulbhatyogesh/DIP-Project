"""Degradation robustness benchmark -- evaluation only, no retraining.

The paper's Conclusion names "different compression levels" as future work.  This
runs that experiment.

Why it matters for THIS method specifically.  Eq. (1) is a single high-pass band:
sigma=3 truncated to 3x3 is nearly a box filter, so the module reads one narrow
slice of the spectrum.  JPEG works by quantising DCT coefficients, and it
quantises the high-frequency ones hardest -- precisely the band the HFC branch
depends on.  Recent work reports that detectors "relying heavily on low-level
features, such as pixel-level artifacts or high-frequency noise patterns, suffer
a more dramatic decrease in performance under JPEG".

Stated before running: we expect the HFC margin over the plain backbone to shrink
as JPEG quality falls, and possibly to invert at low quality.  Whether it does or
not, the degradation curve is the result.

Limitation to report: we degrade the extracted 224x224 crop, which models an image
re-shared through social media.  Re-encoding the source video before face
extraction would be the stricter test.

    python -m src.robustness \
        --ckpt runs/xception_hfc_identity/best.pt \
        --faces-root faces --manifest faces/manifest.csv --splits splits.json \
        --out results/robustness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CelebDFFaces
from src.eval import evaluate
from src.models.build import build_model

# family -> list of (level_label, albumentations op).  `None` is the clean control.
DEGRADATIONS: dict[str, list[tuple[str, object]]] = {
    "clean": [("none", None)],
    "jpeg": [
        (f"q{q}", A.ImageCompression(quality_range=(q, q), p=1.0))
        for q in (90, 70, 50, 30, 20, 10)
    ],
    "noise": [
        (f"s{s}", A.GaussNoise(std_range=(s / 255.0, s / 255.0), mean_range=(0, 0), p=1.0))
        for s in (2, 5, 10)
    ],
    "downscale": [
        (f"x{int(1/f)}", A.Downscale(scale_range=(f, f), p=1.0))
        for f in (0.5, 0.25)
    ],
    "blur": [
        (f"s{s}", A.GaussianBlur(blur_limit=(2 * s + 1, 2 * s + 1), sigma_limit=(s, s), p=1.0))
        for s in (1, 2)
    ],
}


def degraded_transform(op, size: int = 224) -> A.Compose:
    """Evaluation transform with one degradation applied before normalisation."""
    ops = [op] if op is not None else []
    ops += [
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


def measure_psnr(op, dataset: CelebDFFaces, n: int = 24) -> float:
    """Mean PSNR between clean and degraded crops.

    Guards the whole experiment: if a transform is silently a no-op, PSNR stays
    infinite and a null result would be meaningless rather than informative.
    """
    if op is None:
        return float("inf")
    import cv2

    vals = []
    step = max(1, len(dataset) // n)
    for i in range(0, len(dataset), step):
        if len(vals) >= n:
            break
        bgr = cv2.imread(str(dataset.faces_root / dataset.rel_paths[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = A.Compose([op])(image=rgb)["image"]
        mse = float(np.mean((rgb.astype(np.float64) - out.astype(np.float64)) ** 2))
        vals.append(99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse))
    return float(np.mean(vals)) if vals else float("nan")


def subsample_frames(ds: CelebDFFaces, max_per_video: int) -> CelebDFFaces:
    """Keep at most N frames per video, so a sweep of conditions stays affordable."""
    if max_per_video <= 0:
        return ds
    seen: dict[str, int] = {}
    keep: list[int] = []
    for i, vid in enumerate(ds.video_ids):
        c = seen.get(vid, 0)
        if c < max_per_video:
            keep.append(i)
            seen[vid] = c + 1

    ds.rel_paths = [ds.rel_paths[i] for i in keep]
    ds.labels = ds.labels[keep]
    ds.video_ids = [ds.video_ids[i] for i in keep]
    ds.indices_by_label = {}
    for i, y in enumerate(ds.labels):
        ds.indices_by_label.setdefault(int(y), []).append(i)
    return ds


def run_sweep(
    ckpt_path: str | Path,
    faces_root: str | Path,
    manifest: str | Path,
    splits: str | Path,
    families: list[str],
    *,
    part: str = "test",
    max_frames_per_video: int = 8,
    batch_size: int = 64,
    workers: int = 2,
    device: torch.device | None = None,
) -> dict:
    """Score one checkpoint across every requested degradation family."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_model(
        cfg["backbone"], cfg["mode"], pretrained=False,
        kernel_size=cfg.get("kernel_size", 3),
        sigma=cfg.get("sigma", 3.0),
        normalize_kernel=cfg.get("normalize_kernel", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    run = f"{cfg['backbone']}/{cfg['mode']}/{cfg['protocol']}"
    out: dict = {"run": run, "part": part, "iteration": ckpt.get("iteration"), "results": {}}

    for family in families:
        if family not in DEGRADATIONS:
            raise KeyError(f"unknown degradation {family!r}; have {list(DEGRADATIONS)}")
        for level, op in DEGRADATIONS[family]:
            ds = CelebDFFaces(
                faces_root, manifest, splits, cfg["protocol"], part,
                transform=degraded_transform(op), return_meta=True,
            )
            ds = subsample_frames(ds, max_frames_per_video)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=(device.type == "cuda"),
            )
            m = evaluate(model, loader, device)
            m["psnr"] = measure_psnr(op, ds)
            out["results"].setdefault(family, {})[level] = m
            print(
                f"  {run:34s} {family:10s} {level:5s} "
                f"frame_auc={m['frame']['auc']:.4f} "
                f"bal_acc={m['frame']['balanced_accuracy']:.4f} "
                f"psnr={m['psnr']:.1f}",
                flush=True,
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Degradation robustness sweep.")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more best.pt paths")
    ap.add_argument("--faces-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--families", nargs="+", default=["clean", "jpeg", "noise", "downscale", "blur"],
        choices=list(DEGRADATIONS),
    )
    ap.add_argument("--max-frames-per-video", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True, help="directory for the result JSONs")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in args.ckpt:
        print(f"\n===== {ckpt} =====", flush=True)
        res = run_sweep(
            ckpt, args.faces_root, args.manifest, args.splits, args.families,
            part=args.part, max_frames_per_video=args.max_frames_per_video,
            batch_size=args.batch_size, workers=args.workers,
        )
        name = res["run"].replace("/", "_")
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
