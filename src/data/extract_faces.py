"""Extract face crops from Celeb-DF-v2 videos.

Per the paper, Sec. IV-B: 32 frames per sequence, faces located with BlazeFace,
saved as 224x224 RGB images.  Run this ONCE on Kaggle (where celeb-df-v2 is
already mounted, so there is nothing to download) and publish the output as a
private Kaggle dataset that every Colab training session can pull in minutes.

    python -m src.data.extract_faces \
        --root /kaggle/input/celeb-df-v2 \
        --out  /kaggle/working/faces \
        --workers 4

Writes  <out>/<folder>/<video_id>/<frame_idx>.jpg  plus  <out>/manifest.csv,
so that all later split logic is pure metadata and never re-decodes a video.
Safe to re-run: videos already extracted are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.splits import VideoRecord, scan_dataset  # noqa: E402

MANIFEST_FIELDS = [
    "video_id", "folder", "label", "source_id", "target_id", "frame_idx", "rel_path",
]

# BlazeFace weights for the MediaPipe Tasks API.  mediapipe >= 1.0 removed the
# legacy `mp.solutions.face_detection` wrapper, which bundled this model; the
# Tasks API takes it as an explicit .tflite asset instead.
BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_blazeface_model(cache_dir: str | None = None) -> str:
    """Download the BlazeFace .tflite once and return its local path."""
    cache = Path(
        cache_dir
        or os.environ.get("BLAZEFACE_DIR")
        or (Path.home() / ".cache" / "blazeface")
    )
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / "blaze_face_short_range.tflite"
    if not dest.exists():
        import urllib.request

        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(BLAZEFACE_URL, tmp)
        tmp.replace(dest)   # never leave a half-written model behind
    return str(dest)

# One detector per worker process, built lazily (mediapipe objects do not pickle).
_DETECTOR = None
_DETECTOR_KIND = None


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def _build_detector(kind: str):
    """Return a callable frame_rgb -> (x0, y0, x1, y1) or None."""
    if kind == "mediapipe":
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        det = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_blazeface_model()
                ),
                min_detection_confidence=0.5,
            )
        )

        def detect(rgb):
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            res = det.detect(image)
            if not res.detections:
                return None
            best = max(
                res.detections,
                key=lambda d: d.categories[0].score if d.categories else 0.0,
            )
            b = best.bounding_box
            return (
                float(b.origin_x), float(b.origin_y),
                float(b.origin_x + b.width), float(b.origin_y + b.height),
            )

        return detect

    if kind == "mtcnn":
        import torch
        from facenet_pytorch import MTCNN

        det = MTCNN(keep_all=False, device="cuda" if torch.cuda.is_available() else "cpu")

        def detect(rgb):
            boxes, probs = det.detect(rgb)
            if boxes is None or len(boxes) == 0:
                return None
            return tuple(boxes[int(np.argmax(probs))])

        return detect

    raise ValueError(f"unknown detector {kind!r}")


def _get_detector(kind: str):
    global _DETECTOR, _DETECTOR_KIND
    if _DETECTOR is None or _DETECTOR_KIND != kind:
        _DETECTOR = _build_detector(kind)
        _DETECTOR_KIND = kind
    return _DETECTOR


def resolve_detector(kind: str) -> str:
    """Pick a detector, preferring BlazeFace (mediapipe) as the paper specifies."""
    if kind != "auto":
        return kind
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return "mediapipe"
    except ImportError:
        pass
    try:
        import facenet_pytorch  # noqa: F401
        return "mtcnn"
    except ImportError as exc:
        raise RuntimeError(
            "no face detector available; pip install mediapipe (preferred) "
            "or facenet-pytorch"
        ) from exc


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def square_crop(frame: np.ndarray, box, margin: float, size: int) -> np.ndarray:
    """Expand a face box to a square with margin, clamp to frame, resize."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * margin

    half = side / 2.0
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))

    # Pad rather than shrink, so faces near a border keep their aspect ratio.
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(
            frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )
        x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("empty crop")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def sample_frame_indices(n_frames: int, want: int) -> np.ndarray:
    """Evenly spaced indices across the sequence."""
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames <= want:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, want).round().astype(int)


# ---------------------------------------------------------------------------
# Per-video worker
# ---------------------------------------------------------------------------

def extract_one(
    rec_tuple: tuple,
    root: str,
    out: str,
    n_frames: int,
    size: int,
    margin: float,
    detector_kind: str,
    jpeg_quality: int,
) -> dict:
    """Extract up to `n_frames` face crops from one video. Returns a status dict."""
    video_id, folder, label, source_id, target_id, rel_path = rec_tuple
    dest = Path(out) / folder / video_id
    rows: list[dict] = []

    try:
        # Resume: assume a complete directory is already done.
        existing = sorted(dest.glob("*.jpg")) if dest.is_dir() else []
        if len(existing) >= n_frames:
            for p in existing[:n_frames]:
                rows.append({
                    "video_id": video_id, "folder": folder, "label": label,
                    "source_id": source_id or "", "target_id": target_id or "",
                    "frame_idx": int(p.stem), "rel_path": f"{folder}/{video_id}/{p.name}",
                })
            return {"video_id": video_id, "status": "cached", "rows": rows}

        cap = cv2.VideoCapture(str(Path(root) / rel_path))
        if not cap.isOpened():
            return {"video_id": video_id, "status": "unreadable", "rows": []}

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(sample_frame_indices(total, n_frames).tolist())
        if not wanted:
            cap.release()
            return {"video_id": video_id, "status": "empty", "rows": []}

        detect = _get_detector(detector_kind)
        dest.mkdir(parents=True, exist_ok=True)

        idx = 0
        n_missed = 0
        while True:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    box = detect(rgb)
                    if box is None:
                        n_missed += 1
                    else:
                        crop = square_crop(rgb, box, margin, size)
                        name = f"{idx:06d}.jpg"
                        cv2.imwrite(
                            str(dest / name),
                            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                        )
                        rows.append({
                            "video_id": video_id, "folder": folder, "label": label,
                            "source_id": source_id or "", "target_id": target_id or "",
                            "frame_idx": idx, "rel_path": f"{folder}/{video_id}/{name}",
                        })
            idx += 1
        cap.release()

        if not rows:
            return {"video_id": video_id, "status": "no_face", "rows": []}
        status = "ok" if n_missed == 0 else "partial"
        return {"video_id": video_id, "status": status, "rows": rows}

    except Exception:
        return {
            "video_id": video_id,
            "status": "error",
            "rows": [],
            "error": traceback.format_exc(limit=3),
        }


def _to_tuple(r: VideoRecord) -> tuple:
    return (r.video_id, r.folder, r.label, r.source_id, r.target_id, r.rel_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract 224x224 face crops from Celeb-DF-v2.")
    ap.add_argument("--root", required=True, help="Celeb-DF-v2 root (holds Celeb-real/ etc.)")
    ap.add_argument("--out", required=True, help="output directory for crops + manifest.csv")
    ap.add_argument("--frames", type=int, default=32, help="frames per video (paper: 32)")
    ap.add_argument("--size", type=int, default=224, help="crop size (paper: 224)")
    ap.add_argument("--margin", type=float, default=1.3, help="face box expansion factor")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--detector", default="auto", choices=["auto", "mediapipe", "mtcnn"])
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0, help="only N videos (smoke test)")
    ap.add_argument("--include-youtube-real", action="store_true")
    args = ap.parse_args(argv)

    detector_kind = resolve_detector(args.detector)
    if detector_kind == "mediapipe":
        # Fetch the model once here, so N worker processes do not race to
        # download it simultaneously on first use.
        ensure_blazeface_model()

    records = scan_dataset(args.root, include_youtube_real=args.include_youtube_real)
    if args.limit:
        # Keep both classes represented when sampling a smoke-test subset.
        reals = [r for r in records if r.label == 0][: max(1, args.limit // 2)]
        fakes = [r for r in records if r.label == 1][: args.limit - len(reals)]
        records = reals + fakes

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"videos={len(records)}  frames/video={args.frames}  detector={detector_kind}  "
        f"workers={args.workers}  -> {out}",
        flush=True,
    )

    job = dict(
        root=args.root, out=str(out), n_frames=args.frames, size=args.size,
        margin=args.margin, detector_kind=detector_kind, jpeg_quality=args.jpeg_quality,
    )

    rows: list[dict] = []
    counts: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, _to_tuple(r), **job): r.video_id for r in records}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            rows.extend(res["rows"])
            if res["status"] in ("error", "unreadable", "no_face", "empty"):
                failures.append((res["video_id"], res["status"]))
            if i % 100 == 0 or i == len(records):
                done = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"  [{i}/{len(records)}] crops={len(rows)}  {done}", flush=True)

    manifest = out / "manifest.csv"
    rows.sort(key=lambda r: (r["folder"], r["video_id"], r["frame_idx"]))
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if failures:
        log = out / "failures.csv"
        with log.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([("video_id", "status"), *failures])
        print(f"logged {len(failures)} problem videos -> {log}")

    n_videos = len({r["video_id"] for r in rows})
    print(f"\ndone: {len(rows)} crops from {n_videos} videos -> {manifest}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
