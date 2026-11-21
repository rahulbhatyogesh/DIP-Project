"""Build HFC-augmented deepfake detectors on top of timm backbones.

The paper's Table IV parameter counts are reproducible exactly with timm and a
single-output (BCE) head, which is what pins down the architecture:

    legacy_xception,  num_classes=1  -> 20.809 M   (paper: 20.81 M)
    efficientnet_b4,  num_classes=1  -> 17.550 M   (paper: 17.55 M)

Grafting the HFC module then adds ~12 K / ~28 K params, landing on the paper's
20.82 M and 17.58 M.  `verify_param_counts()` asserts all four.

Three setups, matching the rows of Table III:
    "none"      -- unmodified backbone (RGB in).
    "hfc"       -- first conv replaced by HFCModule (colour + high-frequency).
    "hfc_only"  -- Eq. (1) applied per RGB channel and stacked as a 3-channel
                   image fed to an unmodified backbone.  This follows Sec. V-B:
                   "we trained the backbone models with the high-frequency
                   components of each channel, extracted as in (1) and stacked
                   as an RGB image".
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("timm is required: pip install timm") from exc

from .hfc import GaussianHighPass, HFCModule

# Canonical backbone keys -> timm model names.
BACKBONES = {
    "xception": "legacy_xception",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b0": "efficientnet_b0",  # handy for fast smoke tests
}

MODES = ("none", "hfc", "hfc_only")


def _get_module(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class HFCOnlyWrapper(nn.Module):
    """Feed a per-channel high-pass image to an otherwise untouched backbone."""

    def __init__(self, backbone: nn.Module, **hp_kwargs) -> None:
        super().__init__()
        self.highpass = GaussianHighPass(per_channel=True, **hp_kwargs)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.highpass(x))


def build_model(
    backbone: str = "xception",
    mode: str = "hfc",
    *,
    pretrained: bool = True,
    kernel_size: int = 3,
    sigma: float = 3.0,
    normalize_kernel: bool = True,
    filter_type: str = "eq1",
    pyramid_levels: int = 4,
    n_filters: int = 4,
) -> nn.Module:
    """Create a detector.  Output is a single logit per image (use BCEWithLogitsLoss).

    `filter_type` selects the high-pass front end inside the HFC module:
      "eq1"     -- the paper's fixed Gaussian residual (default; reproduces Table IV)
      "pyramid" -- multi-scale Laplacian band-pass
      "learned" -- N filters learned under the constrained-convolution constraint
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    timm_name = BACKBONES.get(backbone, backbone)

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)

    if mode == "none":
        return model

    if mode == "hfc_only":
        # Per-channel Eq. (1) stacked as RGB, fed to an unmodified backbone.
        return HFCOnlyWrapper(
            model, kernel_size=kernel_size, sigma=sigma, normalize_kernel=normalize_kernel
        )

    # mode == "hfc": replace the stem conv in place.
    stem_path = model.default_cfg["first_conv"]
    stem = _get_module(model, stem_path)
    if not isinstance(stem, nn.Conv2d):
        raise TypeError(f"expected {stem_path} to be a Conv2d, got {type(stem).__name__}")

    # C' == the original stem width: in both Xception (bn1 -> act1 -> conv2) and
    # EfficientNet (bn1 -> blocks) the next layer consumes exactly that many channels.
    hfc = HFCModule(
        color_conv=stem,
        out_channels=stem.out_channels,
        kernel_size=kernel_size,
        sigma=sigma,
        normalize_kernel=normalize_kernel,
        filter_type=filter_type,
        pyramid_levels=pyramid_levels,
        n_filters=n_filters,
    )
    _set_module(model, stem_path, hfc)
    return model


def model_from_config(cfg: dict, *, pretrained: bool = False) -> nn.Module:
    """Rebuild a model from a checkpoint's saved config.

    Single source of truth for eval and the robustness sweep -- otherwise a new
    front-end option has to be remembered in three places, and forgetting it
    shows up as an opaque state_dict mismatch.
    """
    return build_model(
        cfg["backbone"],
        cfg["mode"],
        pretrained=pretrained,
        kernel_size=cfg.get("kernel_size", 3),
        sigma=cfg.get("sigma", 3.0),
        normalize_kernel=cfg.get("normalize_kernel", True),
        filter_type=cfg.get("hfc_filter", "eq1"),
        pyramid_levels=cfg.get("pyramid_levels", 4),
        n_filters=cfg.get("n_filters", 4),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# (backbone, mode) -> parameter count in millions, from Table IV of the paper.
PAPER_PARAMS = {
    ("xception", "none"): 20.81,
    ("xception", "hfc"): 20.82,
    ("efficientnet_b4", "none"): 17.55,
    ("efficientnet_b4", "hfc"): 17.58,
}


def verify_param_counts(tol: float = 0.005, verbose: bool = True) -> bool:
    """Assert our models match the paper's Table IV. Strong architecture check."""
    ok = True
    for (backbone, mode), expected in PAPER_PARAMS.items():
        got = count_parameters(build_model(backbone, mode, pretrained=False)) / 1e6
        match = abs(got - expected) <= tol
        ok &= match
        if verbose:
            flag = "OK " if match else "FAIL"
            print(f"  [{flag}] {backbone:16s} {mode:9s} ours={got:7.3f}M  paper={expected:6.2f}M")
    return bool(ok)
