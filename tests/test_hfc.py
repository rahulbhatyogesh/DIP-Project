"""Unit tests for the HFC module and the grafted backbones.

Run:  python -m pytest tests/test_hfc.py -v
      (or)  python tests/test_hfc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.build import (  # noqa: E402
    PAPER_PARAMS,
    build_model,
    count_parameters,
)
from src.models.hfc import GaussianHighPass, HFCModule, gaussian_kernel_2d  # noqa: E402


# --------------------------------------------------------------------------
# Eq. (1): the filter itself
# --------------------------------------------------------------------------

def test_normalised_kernel_sums_to_one():
    k = gaussian_kernel_2d(3, 3.0, normalize=True)
    assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-6)


def test_literal_equation_kernel_sums_to_0148():
    """The paper's printed 1/(2*pi*sigma^2) factor does NOT give a unit-gain blur."""
    k = gaussian_kernel_2d(3, 3.0, normalize=False)
    assert abs(float(k.sum()) - 0.14789) < 1e-4


def test_sigma3_on_3x3_is_nearly_a_box_filter():
    """Corner weight is within 11% of the centre -> essentially uniform."""
    k = gaussian_kernel_2d(3, 3.0, normalize=True)
    assert float(k[0, 0] / k[1, 1]) > 0.89
    assert torch.allclose(k, torch.full_like(k, 1 / 9), atol=0.01)


def test_constant_image_gives_zero_response():
    """A true high-pass must annihilate DC."""
    hp = GaussianHighPass(3, 3.0, normalize_kernel=True)
    out = hp(torch.full((2, 3, 32, 32), 0.6))
    assert out.abs().max() < 1e-5


def test_literal_variant_does_not_annihilate_dc():
    """Contrast case: the literal reading leaves ~85% of the DC behind."""
    hp = GaussianHighPass(3, 3.0, normalize_kernel=False)
    out = hp(torch.full((1, 3, 16, 16), 0.6))
    assert abs(float(out.mean()) - 0.6 * (1 - 0.14789)) < 1e-3


def test_step_edge_gives_antisymmetric_response():
    """A vertical step should produce equal-and-opposite lobes either side."""
    img = torch.zeros(1, 3, 16, 16)
    img[..., 8:] = 1.0
    out = GaussianHighPass(3, 3.0)(img)
    row = out[0, 0, 8]
    assert float(row[7]) < -0.01          # dark side of the edge
    assert float(row[8]) > 0.01           # bright side
    assert abs(float(row[7] + row[8])) < 1e-5   # antisymmetric


def test_highpass_output_channels():
    assert GaussianHighPass(per_channel=False)(torch.rand(1, 3, 16, 16)).shape[1] == 1
    assert GaussianHighPass(per_channel=True)(torch.rand(1, 3, 16, 16)).shape[1] == 3


# --------------------------------------------------------------------------
# The module (Fig. 2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("c_color,expected_hfc", [(32, 10), (48, 16), (64, 21)])
def test_floor_c_over_3_branch_width(c_color, expected_hfc):
    conv = torch.nn.Conv2d(3, c_color, 3, stride=2, bias=False)
    mod = HFCModule(conv, out_channels=c_color)
    assert mod.c_hfc == expected_hfc
    assert mod.fuse.in_channels == c_color + expected_hfc


def test_module_preserves_stem_output_shape():
    """Grafted module must emit exactly what the original stem conv did."""
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, bias=False)
    x = torch.rand(2, 3, 224, 224)
    mod = HFCModule(conv, out_channels=32)
    assert mod(x).shape == conv(x).shape


def test_hfc_only_module_variant_drops_colour_branch():
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, bias=False)
    mod = HFCModule(conv, out_channels=32, use_color=False)
    assert mod.color_conv is None
    assert mod.fuse.in_channels == 10
    assert mod(torch.rand(1, 3, 64, 64)).shape[1] == 32


# --------------------------------------------------------------------------
# Grafted backbones
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backbone,mode,expected", [(b, m, v) for (b, m), v in PAPER_PARAMS.items()])
def test_parameter_counts_match_paper_table_iv(backbone, mode, expected):
    got = count_parameters(build_model(backbone, mode, pretrained=False)) / 1e6
    assert abs(got - expected) <= 0.005, f"{backbone}/{mode}: {got:.3f}M vs paper {expected}M"


@pytest.mark.parametrize("backbone", ["xception", "efficientnet_b4"])
@pytest.mark.parametrize("mode", ["none", "hfc", "hfc_only"])
def test_forward_pass_emits_single_logit(backbone, mode):
    model = build_model(backbone, mode, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_hfc_costs_only_a_few_thousand_params():
    base = count_parameters(build_model("xception", "none", pretrained=False))
    hfc = count_parameters(build_model("xception", "hfc", pretrained=False))
    assert 10_000 < hfc - base < 15_000


def test_gradients_reach_the_hfc_branch():
    """The high-frequency branch must actually train."""
    model = build_model("xception", "hfc", pretrained=False)
    model(torch.rand(2, 3, 224, 224)).sum().backward()
    stem = model.conv1
    assert stem.hfc_conv.weight.grad is not None
    assert stem.hfc_conv.weight.grad.abs().sum() > 0
    assert stem.fuse.weight.grad.abs().sum() > 0


def test_highpass_filter_is_not_learnable():
    """Eq. (1) is a fixed operator, not something the network can drift."""
    model = build_model("xception", "hfc", pretrained=False)
    names = {n for n, _ in model.named_parameters()}
    assert not any("highpass" in n for n in names)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-q"]))
