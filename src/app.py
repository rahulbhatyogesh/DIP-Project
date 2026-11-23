"""Deepfake detector GUI — image and video.

Runs a trained HFC checkpoint on an uploaded file and reports whether it looks
real or manipulated.

Two modes, both using **exactly** the preprocessing the model was trained with
(§3.2 of the report): BlazeFace detection, a 1.3x square margin crop, 224x224,
ImageNet normalisation.  Anything else would score the model off-distribution.

* **Image** — detect the face, score it, show the high-pass residual the HFC
  branch actually reads.
* **Video** — sample 32 evenly spaced frames exactly as `extract_faces.py` does,
  score each, and average the sigmoid outputs into one verdict.  That is the
  same video-level aggregation `src/eval.py` uses for every video-level number
  in the report.

    python -m src.app --ckpt runs/xception_hfc_identity_s42/best.pt

The model outputs P(manipulated); label 1 = fake throughout this project.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.data.extract_faces import (
    _build_detector,
    resolve_detector,
    sample_frame_indices,
    square_crop,
)
from src.models.build import model_from_config

CROP_SIZE = 224
CROP_MARGIN = 1.3          # the paper's 1.3x face margin
N_FRAMES = 32              # the paper's 32 frames per sequence
THRESHOLD = 0.5            # fixed decision threshold on P(manipulated)

# Palette shared with the report figures.
BLUE, RED, INK2, SURFACE, GRID = "#2a78d6", "#e34948", "#52514e", "#fcfcfb", "#e4e3de"

_DETECTOR = None


# --------------------------------------------------------------------- model
def find_checkpoint(explicit: str | None = None) -> Path | None:
    """Locate a checkpoint: the given path, else the newest best.pt nearby."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    roots = [Path("runs"), Path("checkpoints"), Path("models")]
    found = [p for r in roots if r.is_dir() for p in r.rglob("*.pt")]
    if not found:
        return None
    best = [p for p in found if p.name == "best.pt"] or found
    return max(best, key=lambda p: p.stat().st_mtime)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def describe(cfg: dict, ckpt_path: Path) -> str:
    front = cfg.get("hfc_filter", "eq1") if cfg.get("mode") != "none" else "—"
    return (
        f"**Checkpoint** `{ckpt_path.name}`  ·  backbone **{cfg.get('backbone')}**  ·  "
        f"mode **{cfg.get('mode')}**  ·  front end **{front}**  ·  "
        f"split **{cfg.get('protocol')}**  ·  seed {cfg.get('seed', '?')}"
    )


# ----------------------------------------------------------------- detection
def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _build_detector(resolve_detector("auto"))
    return _DETECTOR


def detect_crop(rgb: np.ndarray) -> np.ndarray | None:
    """Face crop at exactly the training geometry, or None if no face."""
    box = get_detector()(rgb)
    if box is None:
        return None
    try:
        return square_crop(rgb, box, CROP_MARGIN, CROP_SIZE)
    except ValueError:
        return None


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    x = np.stack(crops).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def predict(model, crops: list[np.ndarray], device, batch: int = 16) -> np.ndarray:
    """P(manipulated) for each crop."""
    out = []
    for i in range(0, len(crops), batch):
        logits = model(to_tensor(crops[i:i + batch]).to(device))
        out.append(torch.sigmoid(logits.squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------- presentation
def high_pass(crop: np.ndarray) -> np.ndarray:
    """The residual the HFC branch reads — Eq. (1), normalised kernel."""
    g = (crop.astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
    ax = np.arange(3) - 1
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / 18.0)
    hp = g - cv2.filter2D(g, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)
    lim = max(1e-6, float(np.abs(hp).max()) * 0.6)
    return np.clip((hp / lim + 1) / 2, 0, 1)


def verdict_md(p: float, n: int | None = None) -> str:
    fake = p >= THRESHOLD
    label = "MANIPULATED" if fake else "AUTHENTIC"
    colour = RED if fake else "#1baf7a"
    conf = p if fake else 1 - p
    sub = f"P(manipulated) = **{p:.3f}**  ·  decision threshold {THRESHOLD:.2f}"
    if n is not None:
        sub += f"  ·  averaged over **{n}** frames"
    band = ("Close to the threshold — treat as inconclusive."
            if abs(p - THRESHOLD) < 0.15 else "")
    return (
        f"<div style='padding:14px 18px;border-radius:10px;border:2px solid {colour};"
        f"background:{colour}14'>"
        f"<div style='font-size:26px;font-weight:700;color:{colour}'>{label}</div>"
        f"<div style='font-size:15px;margin-top:4px'>confidence {conf:.1%}</div>"
        f"</div>\n\n{sub}" + (f"\n\n*{band}*" if band else "")
    )


def frame_plot(probs: np.ndarray, mean_p: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(probs))
    ax.bar(x, probs, color=[RED if p >= THRESHOLD else BLUE for p in probs], zorder=3)
    ax.axhline(THRESHOLD, color=INK2, lw=1.2, ls=(0, (4, 2.5)), zorder=4)
    ax.axhline(mean_p, color=RED if mean_p >= THRESHOLD else BLUE, lw=2, zorder=5)
    ax.annotate(f"mean {mean_p:.3f}", (len(probs) - 0.5, mean_p), fontsize=8.5,
                ha="right", va="bottom", fontweight="bold",
                color=RED if mean_p >= THRESHOLD else BLUE)
    ax.set_xlabel("sampled frame"), ax.set_ylabel("P(manipulated)")
    ax.set_ylim(0, 1.03), ax.set_xlim(-0.8, len(probs) - 0.2)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0), ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_edgecolor(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    return fig


def contact_sheet(crops: list[np.ndarray], probs: np.ndarray,
                  cols: int = 8) -> np.ndarray:
    """Grid of the scored crops, each bordered by its own verdict."""
    cell, pad = 96, 3
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (cell + 2 * pad), cols * (cell + 2 * pad), 3), 250, np.uint8)
    for i, (c, p) in enumerate(zip(crops, probs)):
        r, col = divmod(i, cols)
        tile = cv2.resize(c, (cell, cell), interpolation=cv2.INTER_AREA)
        bgr = (227, 73, 72) if p >= THRESHOLD else (42, 120, 214)
        tile = cv2.copyMakeBorder(tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                  value=bgr)
        y, x = r * (cell + 2 * pad), col * (cell + 2 * pad)
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return sheet


# ------------------------------------------------------------------ handlers
def analyse_image(img: np.ndarray | None, state):
    model, cfg, device = state
    if img is None:
        return None, None, "Upload an image to begin."
    crop = detect_crop(img)
    if crop is None:
        return None, None, (
            "**No face detected.** This detector is the one the model was trained "
            "with; it expects a reasonably frontal, unoccluded face. The model only "
            "ever saw face crops, so scoring a whole scene would be meaningless."
        )
    p = float(predict(model, [crop], device)[0])
    return crop, high_pass(crop), verdict_md(p)


def analyse_video(path: str | None, state, progress=None):
    model, cfg, device = state
    if not path:
        return None, None, None, "Upload a video to begin."

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None, "**Could not open that video.**"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, N_FRAMES)
    if len(idxs) == 0:
        cap.release()
        return None, None, None, "**That file has no readable frames.**"

    crops, kept, wanted = [], [], set(int(i) for i in idxs)
    pos = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pos in wanted:
            c = detect_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if c is not None:
                crops.append(c)
                kept.append(pos)
            if progress is not None:
                progress(len(crops) / max(1, len(idxs)), desc="detecting faces")
        pos += 1
        if pos > max(wanted):
            break
    cap.release()

    if not crops:
        return None, None, None, (
            f"**No face found in any of the {len(idxs)} sampled frames.** "
            "Nothing to score."
        )

    probs = predict(model, crops, device)
    mean_p = float(probs.mean())
    note = ""
    if len(crops) < len(idxs):
        note = (f"\n\n*{len(idxs) - len(crops)} of {len(idxs)} sampled frames had no "
                f"detectable face and were skipped.*")
    detail = (
        f"\n\nPer-frame spread: min {probs.min():.3f}, max {probs.max():.3f}, "
        f"sd {probs.std(ddof=1) if len(probs) > 1 else 0:.3f}  ·  "
        f"{int((probs >= THRESHOLD).sum())}/{len(probs)} frames above threshold."
    )
    return (
        frame_plot(probs, mean_p),
        contact_sheet(crops, probs),
        None,
        verdict_md(mean_p, n=len(crops)) + detail + note,
    )


# ----------------------------------------------------------------------- UI
INTRO = """
# Deepfake detector — HFC module

Upload a face image or a video. The model is the reproduction from this project:
a backbone whose first convolution is replaced by the paper's High-Frequency
Components module.

**Read the number, not just the word.** A file is called manipulated when
P(manipulated) >= 0.50. The report measures a seed-to-seed standard deviation of
0.0046 AUC and shows the decision threshold collapsing under heavy compression,
so a probability near 0.50 is genuinely inconclusive. This is a course reproduction trained on Celeb-DF-v2 celebrity
faces, not a forensic tool.
"""

NO_CKPT = """
# No checkpoint found

The GUI needs a trained checkpoint. Point it at one:

```bash
python -m src.app --ckpt path/to/best.pt
```

or drop a `best.pt` under `runs/`. Checkpoints are written by `src/train.py`
(see the report's Appendix); they are ~80 MB each and are not kept in the
repository.
"""


def build_ui(state):
    import gradio as gr

    with gr.Blocks(title="Deepfake detector — HFC") as demo:
        gr.Markdown(INTRO)
        gr.Markdown(state[1]["_desc"])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="numpy", label="Upload a face image",
                                      height=300)
                    img_btn = gr.Button("Analyse image", variant="primary")
                with gr.Column(scale=1):
                    img_out = gr.Markdown()
                    with gr.Row():
                        crop_out = gr.Image(label="Face crop the model sees",
                                            height=200)
                        hp_out = gr.Image(label="High-pass residual (Eq. 1)",
                                          height=200)
            img_btn.click(lambda im: analyse_image(im, state),
                          [img_in], [crop_out, hp_out, img_out])

        with gr.Tab("Video"):
            gr.Markdown(
                f"Samples **{N_FRAMES} evenly spaced frames**, crops the face in each "
                "with the same geometry used in training, scores them independently, "
                "and averages the sigmoid outputs — the video-level protocol from the "
                "report."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    vid_in = gr.Video(label="Upload a video", height=300)
                    vid_btn = gr.Button("Analyse video", variant="primary")
                    vid_out = gr.Markdown()
                with gr.Column(scale=1):
                    plot_out = gr.Plot(label="Per-frame P(manipulated)")
                    sheet_out = gr.Image(label="Sampled crops "
                                               "(red = above threshold)")

            def _video(path, progress=gr.Progress()):
                f, s, _, m = analyse_video(path, state, progress)
                return f, s, m

            vid_btn.click(_video, [vid_in], [plot_out, sheet_out, vid_out])

        gr.Markdown(
            "---\n*Label convention: 1 = manipulated. Preprocessing is identical to "
            "training (BlazeFace, 1.3x square margin, 224x224, ImageNet "
            "normalisation), so the scores are in-distribution with the reported "
            "metrics.*"
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", help="path to a trained checkpoint (best.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()

    import gradio as gr

    ckpt = find_checkpoint(a.ckpt)
    if ckpt is None:
        with gr.Blocks(title="Deepfake detector") as demo:
            gr.Markdown(NO_CKPT)
        demo.launch(server_port=a.port, share=a.share, inbrowser=True)
        return

    device = torch.device(a.device)
    print(f"loading {ckpt} on {device} ...")
    model, cfg = load_model(ckpt, device)
    cfg["_desc"] = describe(cfg, ckpt)
    print(cfg["_desc"].replace("**", ""))

    try:
        resolve_detector("auto")
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nThe GUI needs the same face detector as training.")

    import gradio as gr  # noqa: F811  (theme lives on launch() in Gradio 6)

    # On a hosted runtime (Colab) there is no local browser to open, and the
    # only reachable URL is the share link.
    hosted = a.share or "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    build_ui((model, cfg, device)).launch(
        server_port=a.port,
        share=a.share or hosted,
        inbrowser=not hosted,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
