"""Figures for the report.

Generates, from a trained checkpoint and/or raw crops:

  stages      the DIP pipeline on one face -- RGB, grayscale, Gaussian blur,
              high-pass residual -- for the Theory section
  kernels     normalised vs literal Eq. (1) kernels and their responses, which
              is the visual argument for the normalisation choice
  realfake    high-pass residuals of a real and a fake face side by side
  activations Fig. 1 of the paper: HFC-branch activation maps, early vs late
  roc         ROC curves across runs
  tables      Tables II / III / IV assembled from results JSON

    python -m src.figures stages   --faces-root faces --manifest faces/manifest.csv
    python -m src.figures tables   --results results/ --out report/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.hfc import GaussianHighPass, gaussian_kernel_2d  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_manifest(manifest_csv: str | Path) -> list[dict]:
    with Path(manifest_csv).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_face(faces_root, manifest_csv, label: int, index: int = 0) -> np.ndarray:
    """Load one RGB crop of the requested class as float32 in [0, 1]."""
    import cv2

    rows = [r for r in _read_manifest(manifest_csv) if int(r["label"]) == label]
    if not rows:
        raise RuntimeError(f"no crops with label={label} in {manifest_csv}")
    row = rows[index % len(rows)]
    bgr = cv2.imread(str(Path(faces_root) / row["rel_path"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(row["rel_path"])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def _show_signed(ax, arr: np.ndarray, title: str) -> None:
    """Render a signed residual on a symmetric diverging scale."""
    lim = float(np.abs(arr).max()) or 1.0
    im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.axis("off")
    return im


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_stages(img: np.ndarray, out: Path, sigma=3.0, ksize=3) -> Path:
    """RGB -> grayscale -> Gaussian blur -> high-pass residual (Eq. 1)."""
    x = to_tensor(img)
    hp = GaussianHighPass(ksize, sigma, normalize_kernel=True)
    gray = (x * hp.luma).sum(1, keepdim=True)

    import torch.nn.functional as F
    p = ksize // 2
    blur = F.conv2d(F.pad(gray, (p, p, p, p), mode="reflect"), hp.kernel)
    resid = (gray - blur)[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    axes[0].imshow(img); axes[0].set_title("(a) RGB input"); axes[0].axis("off")
    axes[1].imshow(gray[0, 0], cmap="gray")
    axes[1].set_title("(b) grayscale $I_g$"); axes[1].axis("off")
    axes[2].imshow(blur[0, 0], cmap="gray")
    axes[2].set_title(rf"(c) Gaussian blur  $3\times3,\ \sigma={sigma:g}$"); axes[2].axis("off")
    im = _show_signed(axes[3], resid, "(d) $I_{HFC} = I_g - $ blur")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle("Eq. (1): the high-frequency component extraction", y=1.04)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_kernels(img: np.ndarray, out: Path, sigma=3.0, ksize=3) -> Path:
    """Why we normalise the kernel: the literal Eq. (1) is not a high-pass."""
    kn = gaussian_kernel_2d(ksize, sigma, normalize=True)
    kl = gaussian_kernel_2d(ksize, sigma, normalize=False)
    x = to_tensor(img)
    rn = GaussianHighPass(ksize, sigma, True)(x)[0, 0].numpy()
    rl = GaussianHighPass(ksize, sigma, False)(x)[0, 0].numpy()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for row, (k, r, name) in enumerate([
        (kn, rn, f"normalised (sum={float(kn.sum()):.3f})"),
        (kl, rl, f"literal Eq. (1) (sum={float(kl.sum()):.4f})"),
    ]):
        a = axes[row, 0]
        a.imshow(k, cmap="viridis")
        for (i, j), v in np.ndenumerate(k.numpy()):
            a.text(j, i, f"{v:.3f}", ha="center", va="center",
                   color="w", fontsize=7)
        a.set_title(f"kernel -- {name}"); a.axis("off")

        _show_signed(axes[row, 1], r, "response")
        axes[row, 2].hist(r.ravel(), bins=80, color="#4C78A8")
        axes[row, 2].axvline(0, color="k", lw=0.8, ls="--")
        axes[row, 2].set_title(f"response histogram (mean={r.mean():+.3f})")
        axes[row, 2].set_yscale("log")

    fig.suptitle(
        "The printed $1/(2\\pi\\sigma^2)$ factor sums to 0.148 on a $3\\times3$ kernel,\n"
        "so it sharpens rather than high-passes: its response is not zero-mean.",
        y=1.02,
    )
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_realfake(real: np.ndarray, fake: np.ndarray, out: Path,
                 sigma=3.0, ksize=3) -> Path:
    """Real vs fake, in colour and in the high-frequency residual."""
    hp = GaussianHighPass(ksize, sigma, normalize_kernel=True)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for row, (img, name) in enumerate([(real, "REAL"), (fake, "FAKE")]):
        r = hp(to_tensor(img))[0, 0].numpy()
        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{name} -- RGB")
        _show_signed(axes[row, 1], r, f"{name} -- $I_{{HFC}}$")
        axes[row, 2].imshow(np.abs(r), cmap="inferno",
                            vmax=float(np.abs(r).max()) or 1.0)
        axes[row, 2].axis("off")
        axes[row, 2].set_title(f"{name} -- $|I_{{HFC}}|$  (energy={np.abs(r).mean():.4f})")
    fig.suptitle("High-frequency residuals: real vs face-swapped", y=1.02)
    fig.savefig(out)
    plt.close(fig)
    return out


@torch.no_grad()
def fig_activations(ckpt_path: str | Path, real: np.ndarray, fake: np.ndarray,
                    out: Path, n_maps: int = 6) -> Path:
    """Fig. 1 of the paper: what the HFC branch has learned to respond to."""
    from src.models.build import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["backbone"], cfg["mode"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    stem = dict(model.named_modules())[model.default_cfg["first_conv"]]
    if not hasattr(stem, "hfc_conv"):
        raise ValueError("checkpoint is not an HFC model -- nothing to visualise")

    def branch(img):
        x = to_tensor(img)
        h = stem.highpass(x)
        h = stem.hfc_act_in(stem.hfc_bn_in(h))
        h = stem.hfc_conv(h)
        return stem.hfc_act_out(stem.hfc_bn_out(h))[0].numpy()

    acts = {"REAL": branch(real), "FAKE": branch(fake)}
    # Rank channels by how differently they fire on the two classes.
    diff = np.abs(acts["FAKE"].mean((1, 2)) - acts["REAL"].mean((1, 2)))
    picks = np.argsort(diff)[::-1][:n_maps]

    fig, axes = plt.subplots(2, n_maps + 1, figsize=(2.0 * (n_maps + 1), 4.4))
    for row, (name, img) in enumerate([("REAL", real), ("FAKE", fake)]):
        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 0].set_title(name, fontweight="bold")
        for col, c in enumerate(picks, start=1):
            a = axes[row, col]
            a.imshow(acts[name][c], cmap="viridis")
            a.axis("off")
            if row == 0:
                a.set_title(f"ch {c}", fontsize=8)
    fig.suptitle(
        f"HFC-branch activations ({cfg['backbone']}, iteration {ck.get('iteration', '?')}) "
        "-- channels ranked by real/fake separation",
        y=1.03,
    )
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_roc(results: list[dict], out: Path) -> Path:
    """ROC curves for every run, frame level."""
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for r in results:
        fpr, tpr = r.get("roc", (None, None))
        if fpr is None:
            continue
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{r['run']}  AUC={r['frame']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC -- Celeb-DF-v2 test set")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# result tables
# ---------------------------------------------------------------------------

PAPER_TABLE_III = {
    ("xception", "none", "random"): (0.9633, 0.9951),
    ("xception", "hfc", "random"): (0.9663, 0.9963),
    ("xception", "hfc_only", "random"): (0.9431, 0.9901),
    ("xception", "none", "identity"): (0.8834, 0.9669),
    ("xception", "hfc", "identity"): (0.9064, 0.9728),
    ("xception", "hfc_only", "identity"): (0.8750, 0.9665),
    ("efficientnet_b4", "none", "random"): (0.9642, 0.9959),
    ("efficientnet_b4", "hfc", "random"): (0.9654, 0.9950),
    ("efficientnet_b4", "hfc_only", "random"): (0.9198, 0.9742),
    ("efficientnet_b4", "none", "identity"): (0.9056, 0.9756),
    ("efficientnet_b4", "hfc", "identity"): (0.9246, 0.9782),
    ("efficientnet_b4", "hfc_only", "identity"): (0.8123, 0.9171),
}


def build_tables(results_dir: str | Path, level: str = "frame") -> str:
    """Assemble Table III (ours vs paper) from a directory of results JSON."""
    runs: dict[tuple, dict] = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(p.read_text())
        backbone, mode, protocol = r["run"].split("/")
        runs[(backbone, mode, protocol)] = r

    lines = [
        "Table III -- ours vs paper  (level: " + level + ")",
        "",
        f"{'backbone':16s} {'setup':9s} {'split':9s} "
        f"{'acc':>7s} {'(paper)':>8s} {'auc':>7s} {'(paper)':>8s}",
        "-" * 70,
    ]
    for key in PAPER_TABLE_III:
        backbone, mode, protocol = key
        p_acc, p_auc = PAPER_TABLE_III[key]
        r = runs.get(key)
        if r is None:
            lines.append(
                f"{backbone:16s} {mode:9s} {protocol:9s} "
                f"{'--':>7s} {p_acc:>8.4f} {'--':>7s} {p_auc:>8.4f}"
            )
            continue
        m = r[level]
        lines.append(
            f"{backbone:16s} {mode:9s} {protocol:9s} "
            f"{m['accuracy']:>7.4f} {p_acc:>8.4f} {m['auc']:>7.4f} {p_auc:>8.4f}"
        )

    # Table II: the identity-bias contrast, derived from the same runs.
    lines += ["", "Table II -- identity bias (no-HFC rows)", ""]
    for backbone in ("xception", "efficientnet_b4"):
        for protocol in ("random", "identity"):
            r = runs.get((backbone, "none", protocol))
            got = f"{r[level]['accuracy']:.4f} / {r[level]['auc']:.4f}" if r else "-- / --"
            lines.append(f"  {backbone:16s} {protocol:9s}  {got}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate report figures.")
    ap.add_argument("what", choices=[
        "stages", "kernels", "realfake", "activations", "roc", "tables", "all",
    ])
    ap.add_argument("--faces-root", default="faces")
    ap.add_argument("--manifest", default="faces/manifest.csv")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--index", type=int, default=0, help="which crop to use")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--kernel-size", type=int, default=3)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = {args.what} if args.what != "all" else {
        "stages", "kernels", "realfake", "activations", "tables",
    }
    made: list[Path] = []

    needs_face = want & {"stages", "kernels", "realfake", "activations"}
    if needs_face:
        real = load_face(args.faces_root, args.manifest, 0, args.index)
        fake = load_face(args.faces_root, args.manifest, 1, args.index)

    if "stages" in want:
        made.append(fig_stages(real, out / "fig_stages.png", args.sigma, args.kernel_size))
    if "kernels" in want:
        made.append(fig_kernels(real, out / "fig_kernels.png", args.sigma, args.kernel_size))
    if "realfake" in want:
        made.append(fig_realfake(real, fake, out / "fig_realfake.png",
                                 args.sigma, args.kernel_size))
    if "activations" in want:
        if args.ckpt:
            made.append(fig_activations(args.ckpt, real, fake, out / "fig_activations.png"))
        else:
            print("skipping activations: pass --ckpt")
    if "roc" in want:
        results = [json.loads(p.read_text()) for p in Path(args.results).glob("*.json")]
        made.append(fig_roc(results, out / "fig_roc.png"))
    if "tables" in want:
        if Path(args.results).is_dir():
            text = build_tables(args.results)
            (out / "tables.txt").write_text(text)
            print(text)
            made.append(out / "tables.txt")
        else:
            print(f"skipping tables: {args.results} does not exist yet")

    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
