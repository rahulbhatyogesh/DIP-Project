"""Tests for the Dataset, augmentation pipeline and balanced batch sampler.

Builds a tiny on-disk crop set with a deliberately lopsided 1:9 class ratio --
the same imbalance Celeb-DF-v2 has -- so the sampler's job is non-trivial.

Run:  python -m pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (  # noqa: E402
    BalancedBatchSampler,
    CelebDFFaces,
    build_transforms,
    denormalize,
)

N_REAL_VIDEOS, N_FAKE_VIDEOS, FRAMES = 4, 36, 8   # 1:9, like the real dataset


@pytest.fixture(scope="module")
def tiny_set(tmp_path_factory):
    """A miniature extracted-faces directory plus manifest and splits."""
    root = tmp_path_factory.mktemp("tiny")
    faces = root / "faces"
    rows, splits = [], {"train": [], "val": [], "test": []}

    def add(folder, video_id, label, part):
        d = faces / folder / video_id
        d.mkdir(parents=True)
        for f in range(FRAMES):
            img = np.full((224, 224, 3), 60 + 40 * label, np.uint8)
            cv2.imwrite(str(d / f"{f:06d}.jpg"), img)
            rows.append({
                "video_id": video_id, "folder": folder, "label": label,
                "source_id": "id0", "target_id": "id0",
                "frame_idx": f, "rel_path": f"{folder}/{video_id}/{f:06d}.jpg",
            })
        splits[part].append(video_id)

    for i in range(N_REAL_VIDEOS):
        add("Celeb-real", f"id{i}_0000", 0, "train" if i < 3 else "test")
    for i in range(N_FAKE_VIDEOS):
        add("Celeb-synthesis", f"id{i}_id9_0000", 1, "train" if i < 27 else "test")

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    splits_json = root / "splits.json"
    splits_json.write_text(json.dumps({"splits": {"random": splits, "identity": splits}}))
    return faces, manifest, splits_json


@pytest.fixture
def train_ds(tiny_set):
    faces, manifest, splits = tiny_set
    return CelebDFFaces(faces, manifest, splits, "random", "train")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def test_dataset_loads_only_its_own_partition(tiny_set):
    faces, manifest, splits = tiny_set
    train = CelebDFFaces(faces, manifest, splits, "random", "train")
    test = CelebDFFaces(faces, manifest, splits, "random", "test")
    assert len(train) == 30 * FRAMES
    assert len(test) == 10 * FRAMES
    assert not set(train.video_ids) & set(test.video_ids)


def test_item_shape_and_dtype(train_ds):
    img, label = train_ds[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert label.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_return_meta_exposes_video_id(tiny_set):
    faces, manifest, splits = tiny_set
    ds = CelebDFFaces(faces, manifest, splits, "random", "test", return_meta=True)
    img, label, vid = ds[0]
    assert isinstance(vid, str) and img.shape == (3, 224, 224)


def test_dataset_is_imbalanced_as_expected(train_ds):
    c = train_ds.class_counts()
    assert c[1] / c[0] == pytest.approx(9.0)


def test_unknown_protocol_is_rejected(tiny_set):
    faces, manifest, splits = tiny_set
    with pytest.raises(KeyError):
        CelebDFFaces(faces, manifest, splits, "nonsense", "train")


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

def test_train_transform_augments_and_eval_transform_does_not():
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    train_t, eval_t = build_transforms(True), build_transforms(False)
    outs = {train_t(image=img)["image"].sum().item() for _ in range(12)}
    assert len(outs) > 1, "training transform should be stochastic"
    a = eval_t(image=img)["image"]
    b = eval_t(image=img)["image"]
    assert torch.allclose(a, b), "eval transform must be deterministic"


def test_denormalize_round_trips_into_unit_range():
    img = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    t = build_transforms(False)(image=img)["image"]
    back = denormalize(t)
    assert back.min() >= 0.0 and back.max() <= 1.0
    assert torch.allclose(back, torch.from_numpy(img).permute(2, 0, 1) / 255.0, atol=2e-2)


# --------------------------------------------------------------------------
# Balanced sampler -- the paper's "random but balanced" batching
# --------------------------------------------------------------------------

def test_every_batch_is_exactly_half_real_half_fake(train_ds):
    sampler = BalancedBatchSampler(train_ds, batch_size=32, num_batches=40)
    for batch in sampler:
        counts = Counter(int(train_ds.labels[i]) for i in batch)
        assert counts[0] == 16 and counts[1] == 16


def test_sampler_respects_batch_count_and_size(train_ds):
    sampler = BalancedBatchSampler(train_ds, batch_size=32, num_batches=7)
    batches = list(sampler)
    assert len(batches) == 7 and all(len(b) == 32 for b in batches)
    assert len(sampler) == 7


def test_sampler_is_infinite_without_num_batches(train_ds):
    sampler = BalancedBatchSampler(train_ds, batch_size=32)
    it = iter(sampler)
    for _ in range(500):          # far more batches than there is data
        assert len(next(it)) == 32
    with pytest.raises(TypeError):
        len(sampler)


def test_minority_class_is_recycled_not_exhausted(train_ds):
    """Only 24 real frames exist, yet 50 batches need 800 real slots."""
    n_real = train_ds.class_counts()[0]
    sampler = BalancedBatchSampler(train_ds, batch_size=32, num_batches=50)
    drawn = [i for b in sampler for i in b if train_ds.labels[i] == 0]
    assert len(drawn) == 50 * 16
    assert len(set(drawn)) == n_real, "every real frame should get used"


def test_majority_class_is_covered_evenly(train_ds):
    """Fakes are drawn without replacement within a pass, so usage stays flat."""
    sampler = BalancedBatchSampler(train_ds, batch_size=32, num_batches=27)
    drawn = Counter(i for b in sampler for i in b if train_ds.labels[i] == 1)
    assert max(drawn.values()) - min(drawn.values()) <= 1


def test_sampler_is_reproducible(train_ds):
    a = list(BalancedBatchSampler(train_ds, 32, num_batches=5, seed=3))
    b = list(BalancedBatchSampler(train_ds, 32, num_batches=5, seed=3))
    c = list(BalancedBatchSampler(train_ds, 32, num_batches=5, seed=4))
    assert a == b and a != c


def test_odd_batch_size_is_rejected(train_ds):
    with pytest.raises(ValueError):
        BalancedBatchSampler(train_ds, batch_size=31)


def test_single_class_dataset_is_rejected(tiny_set):
    """Guards against silently training on an unusable partition."""
    faces, manifest, splits = tiny_set
    ds = CelebDFFaces(faces, manifest, splits, "random", "train")
    ds.indices_by_label.pop(0)
    with pytest.raises(ValueError):
        BalancedBatchSampler(ds, batch_size=32)


def test_works_with_dataloader(train_ds):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(train_ds, 32, num_batches=3),
        num_workers=0,
    )
    for imgs, labels in loader:
        assert imgs.shape == (32, 3, 224, 224)
        assert labels.sum().item() == 16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
