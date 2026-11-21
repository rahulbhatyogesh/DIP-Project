"""Tests for the two new high-pass front ends.

`LaplacianPyramidHighPass` -- multi-scale band-pass replacing Eq. (1)'s single band.
`ConstrainedHighPass`      -- N learnable filters on the Bayar-Stamm constraint
                              surface, of which Eq. (1) is one fixed member.

The load-bearing tests are perfect reconstruction (the pyramid's defining
property) and that the constraint survives optimisation -- a learnable filter
that drifts off the constraint stops being a high-pass and the whole premise
collapses silently.

Run:  python -m pytest tests/test_frontends.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.build import PAPER_PARAMS, build_model, count_parameters  # noqa: E402
from src.models.hfc import (  # noqa: E402
    FILTER_TYPES,
    ConstrainedHighPass,
    GaussianHighPass,
    HFCModule,
    LaplacianPyramidHighPass,
    build_front_end,
    gaussian_kernel_2d,
)


# --------------------------------------------------------------------------
# Laplacian pyramid
# --------------------------------------------------------------------------

def test_pyramid_perfect_reconstruction():
    """sum of bands + coarsest residual == original. The pyramid's whole point."""
    p = LaplacianPyramidHighPass(levels=4)
    g = torch.rand(2, 1, 224, 224)
    bands, residual = p.decompose(g)
    assert torch.allclose(p.reconstruct(bands, residual), g, atol=1e-5)


def test_pyramid_halves_resolution_each_level():
    p = LaplacianPyramidHighPass(levels=4)
    bands, residual = p.decompose(torch.rand(1, 1, 224, 224))
    assert [b.shape[-1] for b in bands] == [224, 112, 56, 28]
    assert residual.shape[-1] == 14


def test_pyramid_emits_one_channel_per_level():
    for levels in (1, 3, 5):
        p = LaplacianPyramidHighPass(levels=levels)
        out = p(torch.rand(2, 3, 64, 64))
        assert out.shape == (2, levels, 64, 64)
        assert p.out_channels == levels


def test_pyramid_annihilates_dc():
    """Every band is a difference of blurs, so a flat image must give zero."""
    p = LaplacianPyramidHighPass(levels=3)
    assert p(torch.full((1, 3, 64, 64), 0.5)).abs().max() < 1e-5


def test_pyramid_bands_capture_different_scales():
    """A fine checkerboard should excite level 0 far more than the coarsest level."""
    x = torch.zeros(1, 3, 64, 64)
    x[..., ::2, ::2] = 1.0                       # finest possible pattern
    bands, _ = LaplacianPyramidHighPass(levels=4).decompose(x[:, :1])
    energy = [float(b.abs().mean()) for b in bands]
    assert energy[0] > 5 * energy[-1], f"fine detail did not dominate level 0: {energy}"


def test_pyramid_rejects_zero_levels():
    with pytest.raises(ValueError):
        LaplacianPyramidHighPass(levels=0)


# --------------------------------------------------------------------------
# Constrained learnable high-pass
# --------------------------------------------------------------------------

def test_constraint_holds_at_init():
    c = ConstrainedHighPass(n_filters=4, kernel_size=3)
    w = c.constrained_weight().detach()
    for i in range(4):
        assert abs(float(w[i, 0, 1, 1]) - 1.0) < 1e-6
        assert abs(float(w[i].sum() - w[i, 0, 1, 1]) + 1.0) < 1e-6
        assert abs(float(w[i].sum())) < 1e-6          # kernel sums to zero


def test_constraint_survives_optimisation():
    """The point of the parametrisation: gradient steps cannot break the constraint."""
    c = ConstrainedHighPass(n_filters=3, kernel_size=3)
    opt = torch.optim.Adam(c.parameters(), lr=0.1)
    x = torch.rand(4, 3, 32, 32)
    for _ in range(50):
        opt.zero_grad()
        c(x).pow(2).mean().backward()
        opt.step()

    w = c.constrained_weight().detach()
    for i in range(3):
        assert abs(float(w[i, 0, 1, 1]) - 1.0) < 1e-5
        assert abs(float(w[i].sum() - w[i, 0, 1, 1]) + 1.0) < 1e-5
    assert c(torch.full((1, 3, 32, 32), 0.3)).abs().max() < 1e-4, "stopped killing DC"


def test_weights_actually_move():
    """Guards the previous test: a frozen filter would pass it trivially."""
    c = ConstrainedHighPass(n_filters=2, kernel_size=3)
    before = c.constrained_weight().clone()
    opt = torch.optim.Adam(c.parameters(), lr=0.1)
    for _ in range(30):
        opt.zero_grad()
        c(torch.rand(4, 3, 32, 32)).pow(2).mean().backward()
        opt.step()
    assert (c.constrained_weight() - before).abs().max() > 1e-3


def test_filter_zero_is_eq1_rescaled():
    """Filter 0 starts exactly at the paper's configuration, up to centre-1 scaling."""
    c = ConstrainedHighPass(n_filters=1, kernel_size=3, init_sigmas=(3.0,))
    g = gaussian_kernel_2d(3, 3.0, normalize=True)
    expected = -g.clone()
    expected[1, 1] = 1.0 - g[1, 1]
    expected = expected / expected[1, 1]
    assert torch.allclose(c.constrained_weight()[0, 0], expected, atol=1e-6)


def test_constrained_response_matches_eq1_up_to_scale():
    c = ConstrainedHighPass(n_filters=1, kernel_size=3, init_sigmas=(3.0,))
    eq1 = GaussianHighPass(3, 3.0)
    x = torch.rand(2, 3, 48, 48)
    scale = 1.0 / float(1.0 - gaussian_kernel_2d(3, 3.0)[1, 1])
    assert torch.allclose(c(x), eq1(x) * scale, atol=1e-5)


def test_constrained_emits_one_channel_per_filter():
    for n in (1, 4, 8):
        c = ConstrainedHighPass(n_filters=n)
        assert c(torch.rand(2, 3, 32, 32)).shape == (2, n, 32, 32)
        assert c.out_channels == n


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        ConstrainedHighPass(kernel_size=4)


# --------------------------------------------------------------------------
# Integration with HFCModule / build_model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filter_type", list(FILTER_TYPES))
def test_module_accepts_every_front_end(filter_type):
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, bias=False)
    mod = HFCModule(conv, out_channels=32, filter_type=filter_type)
    x = torch.rand(2, 3, 224, 224)
    assert mod(x).shape == conv(x).shape


@pytest.mark.parametrize("filter_type", list(FILTER_TYPES))
def test_backbone_forward_with_every_front_end(filter_type):
    m = build_model("xception", "hfc", pretrained=False, filter_type=filter_type).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 1) and torch.isfinite(out).all()


def test_eq1_remains_the_default_and_matches_the_paper():
    """The new options must not disturb the reproduction."""
    for (backbone, mode), expected in PAPER_PARAMS.items():
        got = count_parameters(build_model(backbone, mode, pretrained=False)) / 1e6
        assert abs(got - expected) <= 0.005, f"{backbone}/{mode} drifted to {got:.3f}M"


def test_new_front_ends_stay_cheap():
    """The selling point is that this costs almost nothing."""
    base = count_parameters(build_model("xception", "hfc", pretrained=False))
    for ft in ("pyramid", "learned"):
        got = count_parameters(build_model("xception", "hfc", pretrained=False, filter_type=ft))
        assert got - base < 3000, f"{ft} added {got - base} params"


def test_unknown_front_end_rejected():
    with pytest.raises(ValueError):
        build_front_end("wavelet")


def test_gradients_reach_learnable_filters():
    m = build_model("xception", "hfc", pretrained=False, filter_type="learned")
    m(torch.rand(2, 3, 224, 224)).sum().backward()
    w = m.conv1.highpass.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# --------------------------------------------------------------------------
# Greyscale control
# --------------------------------------------------------------------------

def test_gray_passthrough_is_plain_luma():
    """No filtering at all: the control for 'is the gain spectral or architectural?'"""
    from src.models.hfc import GrayPassThrough

    g = GrayPassThrough()
    x = torch.rand(2, 3, 32, 32)
    expected = (x * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    assert torch.allclose(g(x), expected, atol=1e-6)
    assert g.out_channels == 1


def test_gray_preserves_dc_unlike_the_high_passes():
    """The point of the control: it keeps exactly what every high-pass removes."""
    from src.models.hfc import GrayPassThrough

    flat = torch.full((1, 3, 16, 16), 0.42)
    assert abs(float(GrayPassThrough()(flat).mean()) - 0.42) < 1e-5
    assert GaussianHighPass(3, 3.0)(flat).abs().max() < 1e-6
