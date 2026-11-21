"""Tests for the degradation robustness benchmark.

The important one is `test_degradations_actually_degrade`: if a transform were
silently a no-op, a flat robustness curve would look like a finding instead of a
bug.  PSNR must fall monotonically with severity in every family.

Run:  python -m pytest tests/test_robustness.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CelebDFFaces  # noqa: E402
from src.robustness import (  # noqa: E402
    DEGRADATIONS,
    degraded_transform,
    measure_psnr,
    subsample_frames,
)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


@pytest.fixture(scope="module")
def face_like() -> np.ndarray:
    """Smooth image with texture -- closer to a face crop than pure noise."""
    y, x = np.mgrid[0:224, 0:224]
    img = (120 + 60 * np.sin(x / 18.0) + 40 * np.cos(y / 25.0)).clip(0, 255).astype(np.uint8)
    return np.stack([img] * 3, -1)


# --------------------------------------------------------------------------
# The degradations themselves
# --------------------------------------------------------------------------

def test_every_degradation_applies_cleanly(face_like):
    for family, levels in DEGRADATIONS.items():
        for level, op in levels:
            if op is None:
                continue
            out = A.Compose([op])(image=face_like)["image"]
            assert out.shape == face_like.shape, f"{family}/{level} changed shape"
            assert out.dtype == np.uint8


@pytest.mark.parametrize("family", ["jpeg", "noise", "downscale", "blur"])
def test_degradations_actually_degrade(face_like, family):
    """PSNR must fall monotonically with severity -- guards against no-op transforms."""
    psnrs = [
        _psnr(face_like, A.Compose([op])(image=face_like)["image"])
        for _, op in DEGRADATIONS[family]
    ]
    assert all(p < 70 for p in psnrs), f"{family} barely changed the image: {psnrs}"
    assert psnrs == sorted(psnrs, reverse=True), f"{family} not monotonic: {psnrs}"


def test_jpeg_quality_spans_a_sensible_range(face_like):
    psnrs = {
        lvl: _psnr(face_like, A.Compose([op])(image=face_like)["image"])
        for lvl, op in DEGRADATIONS["jpeg"]
    }
    assert psnrs["q90"] > 45, "Q90 should be near-lossless"
    assert psnrs["q10"] < psnrs["q90"] - 10, "Q10 should be clearly worse than Q90"


def test_clean_control_is_identity():
    levels = DEGRADATIONS["clean"]
    assert len(levels) == 1 and levels[0][1] is None


# --------------------------------------------------------------------------
# Transform pipeline
# --------------------------------------------------------------------------

def test_degraded_transform_matches_eval_pipeline(face_like):
    t = degraded_transform(DEGRADATIONS["jpeg"][0][1])
    out = t(image=face_like)["image"]
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_degraded_transform_with_none_is_plain_eval(face_like):
    a = degraded_transform(None)(image=face_like)["image"]
    b = degraded_transform(None)(image=face_like)["image"]
    assert torch.allclose(a, b), "clean transform must be deterministic"


def test_measure_psnr_reports_infinite_for_clean():
    assert measure_psnr(None, None) == float("inf")


# --------------------------------------------------------------------------
# Frame subsampling
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("rob")
    faces = root / "faces"
    rows, split = [], []
    for v in range(6):
        vid = f"id{v}_0000"
        d = faces / "Celeb-real" / vid
        d.mkdir(parents=True)
        for f in range(10):
            cv2.imwrite(str(d / f"{f:06d}.jpg"), np.full((224, 224, 3), 100, np.uint8))
            rows.append({
                "video_id": vid, "folder": "Celeb-real", "label": v % 2,
                "source_id": f"id{v}", "target_id": f"id{v}",
                "frame_idx": f, "rel_path": f"Celeb-real/{vid}/{f:06d}.jpg",
            })
        split.append(vid)

    man = root / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = root / "splits.json"
    sp.write_text(json.dumps({"splits": {"random": {"train": [], "val": [], "test": split}}}))
    return faces, man, sp


def test_subsample_keeps_at_most_n_per_video(tiny_set):
    faces, man, sp = tiny_set
    ds = subsample_frames(CelebDFFaces(faces, man, sp, "random", "test"), 3)
    from collections import Counter

    counts = Counter(ds.video_ids)
    assert set(counts.values()) == {3}
    assert len(ds) == 6 * 3


def test_subsample_rebuilds_class_index(tiny_set):
    faces, man, sp = tiny_set
    ds = subsample_frames(CelebDFFaces(faces, man, sp, "random", "test"), 4)
    for label, idx in ds.indices_by_label.items():
        assert all(ds.labels[i] == label for i in idx)
    assert sum(len(v) for v in ds.indices_by_label.values()) == len(ds)


def test_subsample_zero_keeps_everything(tiny_set):
    faces, man, sp = tiny_set
    full = CelebDFFaces(faces, man, sp, "random", "test")
    n = len(full)
    assert len(subsample_frames(full, 0)) == n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
