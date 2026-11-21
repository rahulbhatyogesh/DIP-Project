"""Every hyperparameter the paper specifies, in one place.

Source: Mejri, Papadopoulos & Aouada, Sec. IV-B ("Setup") and Table IV.
Keeping these together means the report's Methodology section can cite a single
file, and that no magic number is buried in the training loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PaperConfig:
    # ---- data (Sec. IV-B) ------------------------------------------------
    frames_per_video: int = 32          # "trained only with 32 frames per sequence"
    image_size: int = 224               # "saved as RGB images of size 224x224"
    face_detector: str = "blazeface"    # "the face tracking module BlazeFace"

    # ---- HFC module (Sec. III-A, Eq. 1) ----------------------------------
    gaussian_kernel: int = 3            # "the Gaussian filter size and sigma are
    gaussian_sigma: float = 3.0         #  set empirically to 3x3 and sigma = 3"
    normalize_kernel: bool = True       # see src/models/hfc.py for why

    # ---- optimisation (Sec. IV-B) ----------------------------------------
    batch_size: int = 32                # "32 images taken in a random but balanced manner"
    learning_rate: float = 1e-3         # "initial learning rate of 10^-3"
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    lr_decay_factor: float = 0.1        # "a decay factor of 0.1"
    lr_patience: int = 10               # "after ten validation passes"
    min_lr: float = 1e-8                # "if the minimum learning rate of 10^-8 is reached"
    max_iterations: int = 100_000       # "trained up to 100k iterations"

    # ---- splits (Sec. V-A) -----------------------------------------------
    test_fraction: float = 0.21         # "21% of the identities for testing"
    val_fraction: float = 0.19          # "and 19% for validation"

    # ---- Table IV: iterations to convergence -----------------------------
    # Used as sensible --max-iters caps; the 100k figure above is only a ceiling.
    convergence_iters: dict[str, int] = field(default_factory=lambda: {
        "efficientnet_b4/none": 40_500,
        "efficientnet_b4/hfc": 36_000,
        "xception/none": 40_000,
        "xception/hfc": 27_000,
    })

    # ---- Table IV: parameter counts (millions) ---------------------------
    paper_params_m: dict[str, float] = field(default_factory=lambda: {
        "efficientnet_b4/none": 17.55,
        "efficientnet_b4/hfc": 17.58,
        "xception/none": 20.81,
        "xception/hfc": 20.82,
    })

    def budget_for(self, backbone: str, mode: str, slack: float = 1.25) -> int:
        """A per-run iteration cap: the paper's convergence point plus headroom."""
        key = f"{backbone}/{'hfc' if mode == 'hfc' else 'none'}"
        return int(self.convergence_iters.get(key, 30_000) * slack)

    def as_dict(self) -> dict:
        return asdict(self)


PAPER = PaperConfig()
