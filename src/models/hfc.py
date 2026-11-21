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
    ) -> None:
        super().__init__()
        self.use_color = use_color

        c_color = color_conv.out_channels
        c_hfc = c_color // 3  # floor(C/3)
        if c_hfc < 1:
            raise ValueError(f"colour branch too narrow ({c_color}) for floor(C/3)")
        self.c_color, self.c_hfc = c_color, c_hfc

        self.highpass = GaussianHighPass(
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_kernel=normalize_kernel,
            per_channel=False,
        )

        # Colour branch: reuse the backbone's conv so pretrained weights carry over.
        self.color_conv = color_conv if use_color else None

        # HFC branch.  Mirrors the colour conv's geometry so the two branches
        # come out at identical spatial resolution and can be concatenated.
        self.hfc_bn_in = nn.BatchNorm2d(1)
        self.hfc_act_in = act_layer(inplace=True)
        self.hfc_conv = nn.Conv2d(
            1,
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
        return f"C={self.c_color}, floor(C/3)={self.c_hfc}, use_color={self.use_color}"
