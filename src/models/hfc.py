"""High-Frequency Components (HFC) module.

Implements Eq. (1) of Mejri, Papadopoulos & Aouada, "Leveraging High-Frequency
Components for Deepfake Detection":

    I_HFC(x,y) = I_g(x,y) - (1/(2*pi*sigma^2)) * sum_{i,j} I_g(x+i, y+j)
                                                 * exp(-(i^2 + j^2) / (2*sigma^2))

i.e. the grayscale image minus a Gaussian-blurred copy of itself -- a high-pass
(unsharp) residual.  Paper settings: 3x3 kernel, sigma = 3.

Two subtleties the paper leaves implicit
----------------------------------------
1.  **Kernel normalisation.**  Eq. (1) carries the *continuous* Gaussian
    normaliser 1/(2*pi*sigma^2).  Applied literally to a 3x3 kernel at sigma=3
    that factor makes the blur term sum to only ~0.148, so the result is
    I_HFC ~= 0.85 * I_g -- a mildly sharpened image, NOT a high-pass.  Fig. 1 of
    the paper plainly shows an edge map, so the intended operator normalises the
    kernel to sum to 1.  `normalize_kernel=True` (default) does that;
    `normalize_kernel=False` reproduces the literal equation for comparison.

2.  **sigma=3 truncated to 3x3 is nearly a box filter.**  The weights come out
    1.000 (centre), 0.946 (edge), 0.895 (corner) -- only ~10% falloff across the
    whole kernel.  The operator is therefore very close to
    I_g - boxblur_3x3(I_g), a discrete Laplacian-style high-pass.  Worth stating
    in the report: it ties the method back to standard DIP sharpening theory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ITU-R BT.601 luma coefficients (the usual RGB -> grayscale conversion).
_LUMA = (0.299, 0.587, 0.114)


def gaussian_kernel_2d(
    kernel_size: int = 3,
    sigma: float = 3.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Return a (kernel_size, kernel_size) Gaussian kernel.

    normalize=True  -> weights sum to 1 (true blur; gives a genuine high-pass).
    normalize=False -> scaled by the continuous 1/(2*pi*sigma^2) factor exactly
                       as printed in Eq. (1) of the paper.
    """
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    if normalize:
        k = k / k.sum()
    else:
        k = k / (2.0 * math.pi * sigma**2)
    return k


class GaussianHighPass(nn.Module):
    """I - gaussian_blur(I).  Fixed (non-learnable) filter.

    per_channel=False : convert RGB -> grayscale first, emit 1 channel (Eq. 1).
    per_channel=True  : apply the filter to R, G and B independently and stack,
                        which is the "HFC only" variant described in Sec. V-B
                        ("the high-frequency components of each channel [...]
                        stacked as an RGB image").
    """

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 3.0,
        normalize_kernel: bool = True,
        per_channel: bool = False,
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.per_channel = per_channel
        self.pad_mode = pad_mode

        k = gaussian_kernel_2d(kernel_size, sigma, normalize_kernel)
        self.register_buffer("kernel", k.view(1, 1, kernel_size, kernel_size))
        self.register_buffer("luma", torch.tensor(_LUMA).view(1, 3, 1, 1))

    @property
    def out_channels(self) -> int:
        return 3 if self.per_channel else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            sig = x
        else:
            sig = (x * self.luma).sum(dim=1, keepdim=True)

        c = sig.shape[1]
        weight = self.kernel.expand(c, 1, -1, -1)
        p = self.kernel_size // 2
        # Reflect-pad so the residual is not dominated by border artefacts.
        blur = F.conv2d(F.pad(sig, (p, p, p, p), mode=self.pad_mode), weight, groups=c)
        return sig - blur


class LaplacianPyramidHighPass(nn.Module):
    """Band-pass decomposition (Burt & Adelson, 1983) in place of Eq. (1)'s single band.

    Motivation.  Eq. (1) extracts ONE narrow high-frequency band -- sigma=3 on a
    3x3 grid is nearly a box filter.  JPEG quantises high-frequency DCT
    coefficients hardest, so that single band is the first thing compression
    destroys.  A pyramid samples several bands, so coarse traces (blending
    boundaries, warping) survive when the finest ones are gone.

        G[0] = I_g,     G[l+1] = reduce(G[l])
        L[l] = G[l] - expand(G[l+1])          band-pass at scale l

    `decompose` returns the bands at native resolution plus the coarsest
    residual, which satisfies perfect reconstruction by construction.  `forward`
    resamples every band back to the input size and stacks them as channels, so
    the HFC branch sees `levels` channels instead of one.

    Uses the classic separable binomial kernel [1,4,6,4,1]/16.
    """

    def __init__(self, levels: int = 4, per_channel: bool = False) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        self.per_channel = per_channel

        k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel", torch.outer(k1, k1).view(1, 1, 5, 5))
        self.register_buffer("luma", torch.tensor(_LUMA).view(1, 3, 1, 1))

    @property
    def out_channels(self) -> int:
        return self.levels * (3 if self.per_channel else 1)

    def _blur(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        c = x.shape[1]
        w = (self.kernel * gain).expand(c, 1, -1, -1)
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), w, groups=c)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._blur(x)[..., ::2, ::2]

    def _expand(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, c, h, w = x.shape
        up = x.new_zeros(b, c, h * 2, w * 2)
        up[..., ::2, ::2] = x                 # zero-insertion upsampling
        up = self._blur(up, gain=4.0)         # gain 4 restores the mean
        return up[..., : size[0], : size[1]]

    def decompose(self, g: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return (band-pass levels at native resolution, coarsest residual)."""
        bands = []
        cur = g
        for _ in range(self.levels):
            nxt = self._reduce(cur)
            bands.append(cur - self._expand(nxt, cur.shape[-2:]))
            cur = nxt
        return bands, cur

    def reconstruct(self, bands: list[torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        """Invert `decompose`. Exact up to floating point -- the pyramid's defining property."""
        cur = residual
        for band in reversed(bands):
            cur = band + self._expand(cur, band.shape[-2:])
        return cur

    def to_gray(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.per_channel else (x * self.luma).sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.to_gray(x)
        bands, _ = self.decompose(g)
        size = g.shape[-2:]
        return torch.cat(
            [F.interpolate(b, size=size, mode="bilinear", align_corners=False) for b in bands],
            dim=1,
        )


class ConstrainedHighPass(nn.Module):
    """N LEARNABLE high-pass filters, generalising the paper's single fixed one.

    Written as one convolution kernel, Eq. (1) is `delta - gaussian`, which has
    centre 0.8804 and off-centre weights summing to -0.8804.  Rescale so the
    centre is 1 and the off-centre weights sum to exactly -1 -- precisely the
    constraint Bayar & Stamm (2016) impose to keep a *learned* first layer
    high-pass, and which the forensics literature has used ever since.

    So the paper hand-picks one member of a family whose members can be learned.
    This module learns `n_filters` of them:

        centre weight        = 1
        sum of all others    = -1      (kernel sums to 0, so DC is annihilated)

    The constraint is re-applied inside `forward`, so it holds for any optimiser
    with no hook in the training loop, and filter 0 is initialised at Eq. (1)
    exactly -- the paper's configuration is the starting point, and any gain is
    attributable to learning rather than to a different initialisation.
    """

    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 3,
        init_sigmas: tuple[float, ...] = (3.0, 1.0, 2.0, 5.0),
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.per_channel = per_channel

        c = kernel_size // 2
        mask = torch.zeros(1, 1, kernel_size, kernel_size)
        mask[0, 0, c, c] = 1.0
        self.register_buffer("centre_mask", mask)
        self.register_buffer("luma", torch.tensor(_LUMA).view(1, 3, 1, 1))

        init = torch.empty(n_filters, 1, kernel_size, kernel_size)
        for i in range(n_filters):
            sigma = init_sigmas[i % len(init_sigmas)]
            g = gaussian_kernel_2d(kernel_size, sigma, normalize=True)
            k = -g
            k[c, c] = 1.0 - g[c, c]
            init[i, 0] = k / k[c, c]          # rescale so the centre is exactly 1
        self.weight = nn.Parameter(init)

    @property
    def out_channels(self) -> int:
        return self.n_filters * (3 if self.per_channel else 1)

    def constrained_weight(self) -> torch.Tensor:
        """Project the raw parameter onto {centre = 1, sum of others = -1}."""
        off = self.weight * (1.0 - self.centre_mask)
        s = off.sum(dim=(-2, -1), keepdim=True)
        safe = torch.where(s.abs() < 1e-8, torch.full_like(s, 1e-8), s)
        return -off / safe + self.centre_mask

    def to_gray(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.per_channel else (x * self.luma).sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.to_gray(x)
        w = self.constrained_weight()
        p = self.kernel_size // 2
        if self.per_channel:                   # apply every filter to every channel
            w = w.repeat(3, 1, 1, 1)
            return F.conv2d(F.pad(g, (p, p, p, p), mode="reflect"), w, groups=3)
        return F.conv2d(F.pad(g, (p, p, p, p), mode="reflect"), w)

    def extra_repr(self) -> str:
        return f"n_filters={self.n_filters}, kernel_size={self.kernel_size}"


class GrayPassThrough(nn.Module):
    """Unfiltered greyscale -- the control the paper never ran.

    Eq. (1) with the literal 1/(2*pi*sigma^2) factor barely high-passes at all
    (it leaves ~85% of the DC through, see `gaussian_kernel_2d`), yet it scores
    within 0.0011 AUC of the proper sum-to-1 high-pass.  That invites an obvious
    question the paper does not answer: is the improvement due to high-frequency
    content, or merely to adding a second input branch?

    This front end removes the filtering entirely and passes I_g straight
    through.  If it matches the HFC module, the benefit is architectural rather
    than spectral, and the paper's stated mechanism does not hold.
    """

    def __init__(self, per_channel: bool = False) -> None:
        super().__init__()
        self.per_channel = per_channel
        self.register_buffer("luma", torch.tensor(_LUMA).view(1, 3, 1, 1))

    @property
    def out_channels(self) -> int:
        return 3 if self.per_channel else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.per_channel else (x * self.luma).sum(dim=1, keepdim=True)


FILTER_TYPES = ("eq1", "pyramid", "learned", "gray")


def build_front_end(
    filter_type: str = "eq1",
    *,
    kernel_size: int = 3,
    sigma: float = 3.0,
    normalize_kernel: bool = True,
    pyramid_levels: int = 4,
    n_filters: int = 4,
    per_channel: bool = False,
) -> nn.Module:
    """Pick the high-pass front end. 'eq1' is the paper, unchanged."""
    if filter_type == "eq1":
        return GaussianHighPass(kernel_size, sigma, normalize_kernel, per_channel)
    if filter_type == "pyramid":
        return LaplacianPyramidHighPass(levels=pyramid_levels, per_channel=per_channel)
    if filter_type == "learned":
        return ConstrainedHighPass(
            n_filters=n_filters, kernel_size=kernel_size, per_channel=per_channel
        )
    if filter_type == "gray":
        return GrayPassThrough(per_channel=per_channel)
    raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")


class HFCModule(nn.Module):
    """Replacement for a backbone's first convolution (Fig. 2 of the paper).

    Two parallel branches:
      * colour branch -- the backbone's own first conv, reused as-is so that
        ImageNet-pretrained weights survive;
      * HFC branch    -- Eq. (1) -> BN -> act -> Conv(1 -> floor(C/3)) -> BN -> act.

    Their outputs are concatenated to C + floor(C/3) channels and fused by a
    3x3 conv emitting `out_channels`, which must match what the backbone's
    second layer expects (in both XceptionNet and EfficientNet that is simply
    the original stem conv's output width).
    """

    def __init__(
        self,
        color_conv: nn.Conv2d,
        out_channels: int,
        *,
        kernel_size: int = 3,
        sigma: float = 3.0,
        normalize_kernel: bool = True,
        use_color: bool = True,
        act_layer: type[nn.Module] = nn.ReLU,
        filter_type: str = "eq1",
        pyramid_levels: int = 4,
        n_filters: int = 4,
    ) -> None:
        super().__init__()
        self.use_color = use_color
        self.filter_type = filter_type

        c_color = color_conv.out_channels
        c_hfc = c_color // 3  # floor(C/3)
        if c_hfc < 1:
            raise ValueError(f"colour branch too narrow ({c_color}) for floor(C/3)")
        self.c_color, self.c_hfc = c_color, c_hfc

        self.highpass = build_front_end(
            filter_type,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_kernel=normalize_kernel,
            pyramid_levels=pyramid_levels,
            n_filters=n_filters,
            per_channel=False,
        )
        n_in = self.highpass.out_channels

        # Colour branch: reuse the backbone's conv so pretrained weights carry over.
        self.color_conv = color_conv if use_color else None

        # HFC branch.  Mirrors the colour conv's geometry so the two branches
        # come out at identical spatial resolution and can be concatenated.
        self.hfc_bn_in = nn.BatchNorm2d(n_in)
        self.hfc_act_in = act_layer(inplace=True)
        self.hfc_conv = nn.Conv2d(
            n_in,
            c_hfc,
            kernel_size=color_conv.kernel_size,
            stride=color_conv.stride,
            padding=color_conv.padding,
            bias=False,
        )
        self.hfc_bn_out = nn.BatchNorm2d(c_hfc)
        self.hfc_act_out = act_layer(inplace=True)

        fuse_in = (c_color if use_color else 0) + c_hfc
        self.fuse = nn.Conv2d(fuse_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.highpass(x)
        h = self.hfc_act_in(self.hfc_bn_in(h))
        h = self.hfc_conv(h)
        h = self.hfc_act_out(self.hfc_bn_out(h))

        if self.use_color:
            h = torch.cat([self.color_conv(x), h], dim=1)
        return self.fuse(h)

    def extra_repr(self) -> str:
        return (
            f"C={self.c_color}, floor(C/3)={self.c_hfc}, "
            f"use_color={self.use_color}, filter={self.filter_type}"
        )
