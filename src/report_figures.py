"""Report figures that need no Celeb-DF data.

Three kinds live here:

* **Schematics** -- the processing pipeline and the HFC module block diagram,
  drawn from the code in `src/models/hfc.py`.
* **Operator demonstrations** -- Eq. (1) and the Laplacian pyramid applied to a
  standard test photograph (`skimage.data.astronaut`), so the intermediate
  stages can be shown without redistributing Celeb-DF frames.
* **Measurement plots** -- the seed study, the front-end comparison, the
  effect-size audit and the degradation sweep, from the recorded numbers.

The figures that need real face crops and trained checkpoints (per-class
activation maps, ROC curves) live in `src/figures.py` and run where the dataset
is mounted.

    python -m src.report_figures --out report/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.patheffects as pe                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter          # noqa: E402

# ---------------------------------------------------------------- design tokens
# Validated subset of the reference categorical palette (light mode, all pairs).
BLUE = "#2a78d6"     # the HFC family / "+ HFC"
ORANGE = "#eb6834"   # the plain backbone / "no HFC"
RED = "#e34948"      # the baseline where it is the odd one out; negative values
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8982"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"

HALO = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
})


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _auc_fmt(v, _=None):
    return "%.3f" % v


def _title(ax, main, sub=None):
    ax.set_title(main, pad=24 if sub else 10)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 11), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5, color=INK2)


# ================================================================ measurements
#                      seed 42,  43,     44     (test frame AUC, Xception/identity)
SEED_AUC = {
    "none":     [0.9895, 0.9923, 0.9834],
    "eq1":      [0.9915, 0.9931, 0.9897],
    "hfc_only": [0.9913, 0.9917, 0.9901],
    "pyramid":  [0.9921, 0.9873, 0.9888],
    "learned":  [0.9916, 0.9897, 0.9886],
}
SEED_BAL = {
    "none":     [0.9503, 0.9533, 0.8804],
    "eq1":      [0.9530, 0.9592, 0.9219],
    "hfc_only": [0.9430, 0.9108, 0.9283],
    "pyramid":  [0.9460, 0.8930, 0.9437],
    "learned":  [0.9565, 0.9385, 0.9217],
}
SEED_LABEL = {
    "none": "none\n(plain backbone)",
    "eq1": "eq1\n(the paper)",
    "hfc_only": "hfc_only",
    "pyramid": "pyramid\n(L=4)",
    "learned": "learned\n(N=4)",
}
ORDER = ["none", "eq1", "hfc_only", "learned", "pyramid"]

CLAIMS = [   # (label, value to print, exact value for the sd multiple)
    ("paper  Xception/random", 0.0012, 0.0012),
    ("paper  B4/identity", 0.0026, 0.0026),
    ("ours  hfc_only (3 seeds)", 0.0026, 0.00263333),
    ("ours  eq1 (3 seeds)", 0.0030, 0.00303333),
    ("paper  Xception/identity", 0.0059, 0.0059),
]

JPEG_Q = ["clean", "Q90", "Q50", "Q30", "Q20", "Q10"]
JPEG_NONE = [0.9764, 0.9710, 0.9654, 0.9522, 0.9076, 0.7154]
JPEG_HFC = [0.9857, 0.9818, 0.9743, 0.9608, 0.9236, 0.7451]

NOISE_L = ["clean", "σ=2", "σ=5", "σ=10"]
NOISE_NONE = [0.9764, 0.9670, 0.9072, 0.7925]
NOISE_HFC = [0.9857, 0.9782, 0.9122, 0.7513]

MARGIN = [
    ("clean", +0.0093), ("JPEG Q90", +0.0108), ("JPEG Q50", +0.0089),
    ("JPEG Q30", +0.0086), ("JPEG Q20", +0.0160), ("JPEG Q10", +0.0297),
    ("blur σ=1", +0.0124), ("noise σ=2", +0.0112), ("noise σ=5", +0.0050),
    ("noise σ=10", -0.0412), ("downscale ×2", +0.0041), ("downscale ×4", -0.0174),
]

BASE_SD = float(np.std(SEED_AUC["none"], ddof=1))


def _stats(d, k):
    v = np.asarray(d[k], dtype=float)
    return v.mean(), v.std(ddof=1)


# ================================================================== schematics
def _box(ax, xy, w, h, text, fc, ec, fontsize=8.2, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, facecolor=fc, edgecolor=ec,
                                zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=3, fontweight=weight, linespacing=1.45)


def _arrow(ax, a, b, color=INK2, style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))


def fig_pipeline(out: Path) -> Path:
    """End-to-end flowchart: video in, per-video decision out."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")

    tint_d, tint_m, tint_e = "#eef4fc", "#fdf0ea", "#eafaf3"

    rows = [
        (0.895, [
            (0.14, "Celeb-DF-v2\n6,220 videos", tint_d, BLUE),
            (0.38, "sample 32 frames\nper video", tint_d, BLUE),
            (0.62, "BlazeFace detect\n+ 1.3× margin crop", tint_d, BLUE),
            (0.86, "197,788 crops\n224×224 JPEG", tint_d, BLUE)]),
        (0.665, [
            (0.14, "split protocol\nrandom / identity", tint_d, BLUE),
            (0.38, "balanced sampler\n16 real + 16 fake", tint_d, BLUE),
            (0.62, "augment\nflip, HSV, contrast", tint_d, BLUE),
            (0.86, "normalise\nImageNet stats", tint_d, BLUE)]),
        (0.415, [
            (0.14, "HFC module\nreplaces conv1", tint_m, ORANGE),
            (0.38, "XceptionNet  /\nEfficientNet-B4", tint_m, ORANGE),
            (0.62, "BCE + Adam\nAMP, T4 GPU", tint_m, ORANGE),
            (0.86, "select best\nval frame AUC", tint_m, ORANGE)]),
        (0.165, [
            (0.14, "test partition\n44,753 crops", tint_e, AQUA),
            (0.38, "frame scores\nsigmoid", tint_e, AQUA),
            (0.62, "video score\nmean over frames", tint_e, AQUA),
            (0.86, "acc · balanced acc\nAUC", tint_e, AQUA)]),
    ]
    w, h = 0.205, 0.115
    for y, items in rows:
        for x, label, fc, ec in items:
            _box(ax, (x, y), w, h, label, fc, ec)
        for i in range(len(items) - 1):
            _arrow(ax, (items[i][0] + w / 2, y), (items[i + 1][0] - w / 2, y))
    # Row-to-row connectors: out of the last box, down the right margin, back
    # along the channel above the next row, then into its first box from above.
    for y0, y1 in ((0.895, 0.665), (0.665, 0.415), (0.415, 0.165)):
        lane = y1 + h / 2 + 0.030
        for a, b in (((0.86 + w / 2, y0), (0.985, y0)),
                     ((0.985, y0), (0.985, lane)),
                     ((0.985, lane), (0.14, lane))):
            _arrow(ax, a, b, style="-", lw=1.2)
        _arrow(ax, (0.14, lane), (0.14, y1 + h / 2), lw=1.2)

    for y, lab, c in ((0.895, "1  DATA", BLUE), (0.665, "2  SAMPLING", BLUE),
                      (0.415, "3  MODEL", ORANGE), (0.165, "4  EVALUATION", AQUA)):
        ax.text(-0.012, y, lab.split("  ")[1], fontsize=7.6, color=c,
                fontweight="bold", ha="center", va="center", rotation=90)

    ax.text(0.0, 1.035, "Processing pipeline", fontsize=11.5, fontweight="bold",
            color=INK, ha="left", va="center")
    fig.tight_layout()
    p = out / "pipeline.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_module(out: Path) -> Path:
    """The HFC module itself, exactly as src/models/hfc.py builds it."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")

    _box(ax, (0.075, 0.5), 0.12, 0.17, "RGB\n224×224×3", "#f3f2ee", MUTED)

    # colour branch (top)
    _box(ax, (0.34, 0.76), 0.25, 0.17,
         "Conv 3×3, stride 2\n→ C channels\n(pretrained conv1)", "#fdf0ea", ORANGE)
    # hfc branch (bottom)
    _box(ax, (0.265, 0.235), 0.20, 0.17,
         "high-pass\nfront end\n→ n channels", "#eef4fc", BLUE)
    _box(ax, (0.50, 0.235), 0.20, 0.17,
         "BN → ReLU →\nConv 3×3, stride 2\n→ ⌊C/3⌋ channels", "#eef4fc", BLUE)

    _box(ax, (0.735, 0.5), 0.115, 0.17, "concat\nC + ⌊C/3⌋", "#f3f2ee", MUTED)
    _box(ax, (0.915, 0.5), 0.13, 0.17, "fuse\nConv 3×3\n→ C out", "#f3f2ee", INK2)

    _arrow(ax, (0.135, 0.545), (0.215, 0.74))
    _arrow(ax, (0.135, 0.455), (0.165, 0.26))
    _arrow(ax, (0.365, 0.235), (0.40, 0.235))
    _arrow(ax, (0.465, 0.74), (0.677, 0.55))
    _arrow(ax, (0.60, 0.235), (0.677, 0.45))
    _arrow(ax, (0.793, 0.5), (0.85, 0.5))
    _arrow(ax, (0.98, 0.5), (1.03, 0.5))
    ax.text(1.04, 0.5, "to backbone\nblock 2", fontsize=8, color=INK2, va="center")

    ax.text(0.34, 0.90, "colour branch", fontsize=8.5, color=ORANGE,
            fontweight="bold", ha="center")
    ax.text(0.38, 0.085, "high-frequency branch", fontsize=8.5, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(0.265, 0.045, "eq1 · pyramid · learned · gray",
            fontsize=7.6, color=INK2, ha="center", style="italic")

    ax.text(0.0, 1.10, "The HFC module replaces the backbone's first convolution",
            fontsize=11.5, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.0, 1.00, "C = 32 for XceptionNet, 48 for EfficientNet-B4; "
                       "setting use_color=False gives the hfc_only ablation",
            fontsize=8.5, color=INK2, ha="left", va="center")
    fig.tight_layout()
    p = out / "hfc_module.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


# ====================================================== operator demonstrations
def _test_face() -> np.ndarray:
    """A 224x224 face crop from a standard test photograph (not Celeb-DF)."""
    from skimage import data
    from skimage.transform import resize
    img = data.astronaut()[20:300, 150:400]        # head and shoulders
    return resize(img, (224, 224), anti_aliasing=True).astype(np.float32)


def _gray(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _gauss3(sigma=3.0, k=3):
    ax_ = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax_, ax_)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def fig_stages(out: Path) -> Path:
    """Intermediate stages of Eq. (1) on a real photograph."""
    import cv2
    rgb = _test_face()
    g = _gray(rgb)
    blur = cv2.filter2D(g, -1, _gauss3(), borderType=cv2.BORDER_REFLECT)
    hp = g - blur

    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.5))
    panels = [
        (rgb, "1.  input RGB", None, None),
        (g, "2.  greyscale $I_g$", "gray", (0, 1)),
        (blur, "3.  blur (3×3, σ=3)", "gray", (0, 1)),
        (hp, "4.  $I_{HFC} = I_g -$ blur", "gray", (-0.12, 0.12)),
        (None, "5.  log |FFT| of $I_{HFC}$", "magma", None),
    ]
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(hp))))
    panels[4] = (spec, panels[4][1], "magma", (spec.min(), spec.max() * 0.92))

    for ax, (im, t, cmap, lim) in zip(axes, panels):
        if cmap is None:
            ax.imshow(np.clip(im, 0, 1))
        else:
            ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_title(t, fontsize=8.6, pad=6)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    axes[3].text(0.5, -0.07, "mean %.2e — DC removed" % hp.mean(),
                 transform=axes[3].transAxes, ha="center", va="top",
                 fontsize=7.6, color=BLUE)
    fig.suptitle("Intermediate stages of the high-pass front end",
                 x=0.005, y=1.10, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.text(0.005, 1.005, "standard test photograph, not a Celeb-DF frame; "
             "the residual keeps edges, pores and compression texture and "
             "discards flat skin",
             ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout()
    p = out / "stages.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pyramid(out: Path) -> Path:
    """Laplacian bands, and what JPEG does to each of them."""
    import cv2
    g = _gray(_test_face())

    def bands(img, levels=4):
        cur, out_ = img.copy(), []
        for _ in range(levels):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            out_.append(cur - up)
            cur = down
        return out_, cur

    ok, enc = cv2.imencode(".jpg", (g * 255).clip(0, 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    assert ok
    gj = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    b_clean, res = bands(g)
    b_jpeg, _ = bands(gj)

    # Burt-Adelson perfect reconstruction, as a correctness check.
    rec = res.copy()
    for L in reversed(b_clean):
        rec = cv2.pyrUp(rec, dstsize=(L.shape[1], L.shape[0])) + L
    recon_err = float(np.abs(rec - g).max())

    corr = [float(np.corrcoef(b_clean[i].ravel(), b_jpeg[i].ravel())[0, 1])
            for i in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(8.4, 5.0))
    for lev in range(4):
        lim = max(0.02, np.abs(b_clean[lev]).max() * 0.55)

        ax = axes[0, lev]
        ax.imshow(b_clean[lev], cmap="gray", vmin=-lim, vmax=lim)
        ax.set_title("$L_%d$   %d×%d" % (lev, b_clean[lev].shape[0],
                                         b_clean[lev].shape[1]),
                     fontsize=8.8, pad=6)
        ax.text(0.5, -0.06, "RMS %.4f" % np.sqrt((b_clean[lev] ** 2).mean()),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=INK2)

        ax = axes[1, lev]
        d = np.abs(b_clean[lev] - b_jpeg[lev])
        ax.imshow(d, cmap="inferno", vmin=0, vmax=lim)
        ax.text(0.5, -0.06, "corr with clean  %.3f" % corr[lev],
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color=RED if corr[lev] < 0.95 else INK2,
                fontweight="bold" if corr[lev] < 0.95 else "normal")

        for ax in (axes[0, lev], axes[1, lev]):
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(GRID)

    axes[0, 0].set_ylabel("band", fontsize=9, color=INK2, fontweight="bold")
    axes[1, 0].set_ylabel("changed by\nJPEG Q20", fontsize=9, color=RED,
                          fontweight="bold")

    fig.suptitle("Coarse bands survive compression; the finest one is rewritten",
                 x=0.005, y=1.145, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.text(0.005, 1.105,
             "Top: the four Laplacian bands of the crop, reconstructing it to "
             "%.1e. Bottom: what JPEG Q20 changed in each. Band energy barely "
             "moves, but $L_0$ keeps only %.0f%% correlation with its clean "
             "self — compression does not drain the finest band, it replaces "
             "it with 8×8 block artefacts. By $L_3$ the band is untouched "
             "(%.3f)." % (recon_err, 100 * corr[0], corr[3]),
             ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(h_pad=2.6)
    p = out / "pyramid_bands.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_eq1_spectrum(out: Path) -> Path:
    """Eq. (1) as printed vs as it must be read -- weights and spectrum."""
    sigma, k = 3.0, 3
    ax_r = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax_r, ax_r)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    literal, proper = g, g / g.sum()
    delta = np.zeros((k, k))
    delta[k // 2, k // 2] = 1.0
    hp_lit, hp_prop = delta - literal, delta - proper

    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.4], hspace=0.62, wspace=0.40,
                          left=0.035, right=0.975, top=0.80, bottom=0.21)

    for row, (K, name, dc) in enumerate([
            (hp_lit, "as printed  (normaliser sums to %.4f)" % literal.sum(),
             1 - literal.sum()),
            (hp_prop, "normalised  (the reading we adopt)", 1 - proper.sum())]):
        ax = fig.add_subplot(gs[row, 0])
        lim = np.abs(K).max()
        ax.imshow(K, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, "%+.3f" % K[i, j], ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(K[i, j]) > 0.55 * lim else INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("%s\nDC gain = %.3f" % (name, dc), fontsize=8.5)

        ax = fig.add_subplot(gs[row, 1])
        H = np.abs(np.fft.fftshift(np.fft.fft2(K, s=(64, 64))))
        im = ax.imshow(H, cmap="magma", extent=(-0.5, 0.5, -0.5, 0.5), vmin=0, vmax=2)
        ax.set_xticks([-0.5, 0, 0.5]), ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel("$f_x$ (cycles/px)", fontsize=8)
        ax.set_title("magnitude response", fontsize=8.5)

    cax = fig.add_axes([0.285, 0.075, 0.15, 0.022])
    fig.colorbar(im, cax=cax, orientation="horizontal").set_label("|H|", fontsize=8)
    cax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[:, 2])
    f = np.fft.fftshift(np.fft.fftfreq(256))
    for K, lab, c in ((hp_lit, "as printed", ORANGE), (hp_prop, "normalised", BLUE)):
        H = np.abs(np.fft.fftshift(np.fft.fft2(K, s=(256, 256))))[128, :]
        ax.plot(f, H, color=c, lw=2, label=lab, zorder=3)
    for val, c in ((1 - literal.sum(), ORANGE), (0.0, BLUE)):
        ax.plot(0, val, "o", color=c, ms=8, mec=SURFACE, mew=1.5, zorder=6)
    ax.annotate("$|H(0)|$ = %.3f" % (1 - literal.sum()), (-0.485, 0.175),
                color=ORANGE, fontsize=8.4, fontweight="bold", ha="left")
    ax.annotate("$|H(0)|$ = 0.000", (-0.485, 0.055),
                color=BLUE, fontsize=8.4, fontweight="bold", ha="left")
    ax.set_xlabel("spatial frequency (cycles/px)")
    ax.set_xlim(-0.5, 0.5), ax.set_ylim(-0.05, 1.52)
    ax.set_title("Horizontal slice through $|H|$", fontsize=9.5)
    ax.legend(loc="upper center")
    ax.grid(zorder=0), ax.set_axisbelow(True)
    _despine(ax)

    fig.suptitle("Eq. (1) as printed is not a high-pass filter",
                 x=0.012, y=0.975, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.text(0.012, 0.925, "the printed normaliser sums to 0.1479, not 1 — so the "
             "subtraction sharpens rather than high-passes",
             ha="left", fontsize=8.5, color=INK2)
    p = out / "eq1_spectrum.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


# ============================================================ measurement plots
def fig_seed_floor(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for i, k in enumerate(ORDER):
        v = np.array(SEED_AUC[k])
        c = RED if k == "none" else BLUE
        ax.plot([i, i], [v.min(), v.max()], color=c, lw=2, alpha=0.30,
                solid_capstyle="round", zorder=1)
        ax.plot([i] * 3, v, "o", ms=7, mfc=c, mec=SURFACE, mew=1.4, zorder=3)
        ax.plot([i - 0.21, i + 0.21], [v.mean()] * 2, color=c, lw=2.4,
                solid_capstyle="round", zorder=4)
        ax.annotate("%.4f" % v.mean(), (i + 0.26, v.mean()), fontsize=8,
                    color=INK, va="center", ha="left", fontweight="bold")

    nb = np.array(SEED_AUC["none"])
    ax.axhspan(nb.min(), nb.max(), color=RED, alpha=0.055, zorder=0)
    ax.annotate("the baseline's own 3-seed spread: 0.0089 AUC —\n"
                "wider than every difference between these means",
                (0.55, nb.min() + 0.0004), fontsize=8,
                color=RED, va="bottom", ha="left", style="italic")

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SEED_LABEL[k] for k in ORDER])
    ax.set_ylabel("test frame AUC")
    ax.set_xlim(-0.55, 5.15)
    ax.yaxis.set_major_formatter(FuncFormatter(_auc_fmt))
    ax.grid(axis="y", zorder=0), ax.set_axisbelow(True)
    _despine(ax)
    _title(ax, "Three seeds of the identical configuration",
           "Xception, identity split — only the random seed differs")
    fig.tight_layout()
    p = out / "seed_floor.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig_frontends(out: Path) -> Path:
    """Front ends compared on both metrics, with seed uncertainty."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    names = ["eq1", "learned", "pyramid", "hfc_only", "none"]
    short = {"eq1": "eq1 (paper)", "learned": "learned (N=4)",
             "pyramid": "pyramid (L=4)", "hfc_only": "hfc_only",
             "none": "none"}

    # Dots, not bars: the interesting range does not include zero, and a bar
    # chart on a truncated axis would exaggerate the differences.
    for ax, (src, lab, pad) in zip(axes, ((SEED_BAL, "balanced accuracy", 0.013),
                                          (SEED_AUC, "frame AUC", 0.0011))):
        y = np.arange(len(names))[::-1]
        m = np.array([_stats(src, k)[0] for k in names])
        s = np.array([_stats(src, k)[1] for k in names])
        cols = [RED if k == "none" else BLUE for k in names]

        # the baseline's interval, carried across as the resolution limit
        i0 = names.index("none")
        ax.axvspan(m[i0] - s[i0], m[i0] + s[i0], color=RED, alpha=0.07, zorder=0)

        for yi, mi, si, c in zip(y, m, s, cols):
            ax.errorbar(mi, yi, xerr=si, fmt="none", ecolor=c, elinewidth=2,
                        capsize=5, capthick=1.6, alpha=0.85, zorder=3)
            ax.plot(mi, yi, "o", ms=9, mfc=c, mec=SURFACE, mew=1.5, zorder=4)
            ax.annotate("%.4f ± %.4f" % (mi, si), (mi, yi + 0.30), fontsize=7.8,
                        va="bottom", ha="center", color=INK, zorder=6,
                        path_effects=HALO)

        ax.set_yticks(y)
        ax.set_yticklabels([short[k] for k in names])
        ax.set_xlabel("%s  (mean ± 1 sd over 3 seeds)" % lab)
        lo, hi = (m - s).min(), (m + s).max()
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(names) - 0.25)
        ax.grid(axis="x", zorder=0), ax.set_axisbelow(True)
        _despine(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)

    axes[0].set_title("Balanced accuracy", fontsize=9.5)
    axes[1].set_title("Frame AUC", fontsize=9.5)
    fig.suptitle("No front end separates from the others once seeds are counted",
                 x=0.006, y=1.045, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.text(0.006, 0.965, "Xception, identity split, test partition. "
             "Error bars are one standard deviation over seeds 42, 43 and 44; "
             "every pair of intervals overlaps.",
             ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout()
    p = out / "frontends.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_effect_sizes(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    names = [c[0] for c in CLAIMS]
    vals = [c[1] for c in CLAIMS]
    exact = [c[2] for c in CLAIMS]
    y = np.arange(len(vals))

    ax.barh(y, vals, height=0.56, zorder=3,
            color=[BLUE if "ours" in n else ORANGE for n in names])
    for i, (v, e) in enumerate(zip(vals, exact)):
        ax.annotate("%+.4f   (%.2f sd)" % (v, e / BASE_SD), (v + 0.00024, i),
                    va="center", fontsize=8, color=INK, zorder=6,
                    path_effects=HALO)

    top = len(vals) - 0.42
    ax.axvline(BASE_SD, color=RED, lw=1.6, ls=(0, (4, 2.5)), zorder=4)
    ax.axvline(2 * BASE_SD, color=RED, lw=2, zorder=4)
    ax.annotate("1 sd", (BASE_SD - 0.00022, top), color=RED, fontsize=8,
                ha="right", va="bottom")
    ax.annotate("2 sd = the resolution limit\nnothing reaches it",
                (2 * BASE_SD + 0.00032, top), color=RED, fontsize=8.5,
                ha="left", va="bottom", fontweight="bold")

    ax.set_yticks(y), ax.set_yticklabels(names)
    ax.set_xlabel("claimed gain in frame AUC")
    ax.set_xlim(0, 0.0138)
    ax.set_ylim(-0.6, len(vals) + 1.05)
    ax.grid(axis="x", zorder=0), ax.set_axisbelow(True)
    _despine(ax)
    _title(ax, "Every claimed effect sits inside the seed noise",
           "orange = the paper's claims,  blue = ours")
    fig.tight_layout()
    p = out / "effect_sizes.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig_degradation(out: Path) -> Path:
    fig = plt.figure(figsize=(9.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.92, 1.3], wspace=0.40,
                          left=0.065, right=0.985, top=0.735, bottom=0.145)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    ax = axes[0]
    x = np.arange(len(JPEG_Q))
    ax.plot(x, JPEG_NONE, "-o", color=ORANGE, lw=2, ms=7, mec=SURFACE, mew=1.3,
            label="no HFC", zorder=3)
    ax.plot(x, JPEG_HFC, "-o", color=BLUE, lw=2, ms=7, mec=SURFACE, mew=1.3,
            label="+ HFC", zorder=3)
    ax.fill_between(x, JPEG_NONE, JPEG_HFC, color=BLUE, alpha=0.13, zorder=1)
    ax.annotate("margin +0.0297", (4.92, 0.806), fontsize=8.5, color=BLUE,
                ha="left", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
                xytext=(0.12, 0.728))
    ax.set_xticks(x), ax.set_xticklabels(JPEG_Q)
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("test frame AUC")
    ax.set_title("JPEG $\\it{removes}$ high-frequency energy\nthe margin widens",
                 fontsize=9.5)
    ax.legend(loc="upper right")

    ax = axes[1]
    x = np.arange(len(NOISE_L))
    ax.plot(x, NOISE_NONE, "-o", color=ORANGE, lw=2, ms=7, mec=SURFACE, mew=1.3,
            label="no HFC", zorder=3)
    ax.plot(x, NOISE_HFC, "-o", color=BLUE, lw=2, ms=7, mec=SURFACE, mew=1.3,
            label="+ HFC", zorder=3)
    n_, h_ = np.array(NOISE_NONE), np.array(NOISE_HFC)
    ax.fill_between(x, n_, h_, where=h_ >= n_, color=BLUE, alpha=0.13,
                    zorder=1, interpolate=True)
    ax.fill_between(x, n_, h_, where=h_ <= n_, color=RED, alpha=0.18,
                    zorder=1, interpolate=True)
    ax.annotate("margin −0.0412", (2.93, 0.773), fontsize=8.5, color=RED,
                ha="left", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                xytext=(0.06, 0.724))
    ax.set_xticks(x), ax.set_xticklabels(NOISE_L)
    ax.set_xlabel("additive Gaussian noise")
    ax.set_title("Noise $\\it{adds}$ high-frequency energy\nthe margin inverts",
                 fontsize=9.5)
    ax.legend(loc="upper right")

    for ax in axes:
        ax.set_ylim(0.70, 1.002)
        ax.yaxis.set_major_formatter(FuncFormatter(_auc_fmt))
        ax.grid(axis="y", zorder=0), ax.set_axisbelow(True)
        _despine(ax)

    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(MARGIN))[::-1]
    vals = [m[1] for m in MARGIN]
    ax.barh(y, vals, height=0.62, zorder=3,
            color=[BLUE if v > 0 else RED for v in vals])
    for yi, v in zip(y, vals):
        ax.annotate("%+.4f" % v, (v + (0.0013 if v > 0 else -0.0013), yi),
                    va="center", ha="left" if v > 0 else "right",
                    fontsize=7.8, color=INK,
                    fontweight="bold" if abs(v) > 0.017 else "normal")
    ax.axvline(0, color=INK2, lw=1.1, zorder=4)
    ax.set_yticks(y), ax.set_yticklabels([m[0] for m in MARGIN], fontsize=8)
    ax.set_xlabel("HFC margin  (+HFC − no HFC), frame AUC")
    ax.set_xlim(-0.072, 0.052)
    ax.grid(axis="x", zorder=0), ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.set_title("Margin by condition", fontsize=9.5, pad=8)

    fig.suptitle("The degradation family decides the sign of the HFC margin",
                 x=0.012, y=0.99, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    p = out / "degradation.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "ranking_reversal.png"
    if stale.exists():
        stale.unlink()
        print("removed stale", stale)
    for fn in (fig_pipeline, fig_module, fig_eq1_spectrum, fig_stages,
               fig_pyramid, fig_seed_floor, fig_frontends, fig_effect_sizes,
               fig_degradation):
        p = fn(out)
        print("wrote %-40s %6.1f KB" % (p, p.stat().st_size / 1024))


if __name__ == "__main__":
    main()
