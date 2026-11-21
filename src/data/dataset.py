"""Dataset, augmentation and the balanced batch sampler.

Sec. IV-B of the paper: faces are 224x224 RGB; Albumentations supplies "random
horizontal flipping, random hue, saturation, brightness, and contrast changes";
and "the batch size is 32 images taken in a random but balanced manner".

That last clause matters.  Celeb-DF-v2 is ~1:9.5 real:fake, so plain shuffling
would feed batches that are ~90% fake and the resulting accuracy would be close
to meaningless.  `BalancedBatchSampler` emits exactly half real / half fake per
batch.

Training is counted in ITERATIONS, not epochs (the paper quotes 27k-40k
iterations to convergence in Table IV), so the sampler is infinite by design.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, size: int = 224, normalize: bool = True) -> A.Compose:
    """Paper's augmentation set for training; plain normalisation otherwise."""
    ops: list = []
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
        ]
    # Crops are written at `size` already; this is a cheap safety net.
    ops.append(A.Resize(size, size))
    if normalize:
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        ops.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))
    ops.append(ToTensorV2())
    return A.Compose(ops)


class CelebDFFaces(Dataset):
    """Face crops for one (protocol, partition) pair.

    protocol : "random" | "identity"
    part     : "train"  | "val" | "test"
    """

    def __init__(
        self,
        faces_root: str | Path,
        manifest_csv: str | Path,
        splits_json: str | Path,
        protocol: str,
        part: str,
        *,
        transform: A.Compose | None = None,
        size: int = 224,
        normalize: bool = True,
        return_meta: bool = False,
    ) -> None:
        self.faces_root = Path(faces_root)
        self.protocol, self.part = protocol, part
        self.return_meta = return_meta
        self.transform = transform or build_transforms(
            train=(part == "train"), size=size, normalize=normalize
        )

        splits = json.loads(Path(splits_json).read_text())
        if protocol not in splits["splits"]:
            raise KeyError(f"unknown protocol {protocol!r}; have {list(splits['splits'])}")
        keep = set(splits["splits"][protocol][part])

        with Path(manifest_csv).open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["video_id"] in keep]
        if not rows:
            raise RuntimeError(f"no frames for {protocol}/{part} -- was extraction run?")

        self.rel_paths = [r["rel_path"] for r in rows]
        self.labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        self.video_ids = [r["video_id"] for r in rows]

        # Index by class once, for the balanced sampler.
        self.indices_by_label: dict[int, list[int]] = {}
        for i, y in enumerate(self.labels):
            self.indices_by_label.setdefault(int(y), []).append(i)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, i: int):
        path = self.faces_root / self.rel_paths[i]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"unreadable crop: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.transform(image=rgb)["image"]
        label = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        if self.return_meta:
            return img, label, self.video_ids[i]
        return img, label

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(int(y) for y in self.labels))

    def describe(self) -> str:
        c = self.class_counts()
        n_vid = len(set(self.video_ids))
        return (
            f"{self.protocol}/{self.part}: {len(self)} frames from {n_vid} videos "
            f"(real={c.get(0, 0)}, fake={c.get(1, 0)})"
        )


class BalancedBatchSampler(Sampler[list[int]]):
    """Infinite sampler yielding batches with equal numbers of each class.

    Implements the paper's "random but balanced" batching.  Each class is drawn
    without replacement and reshuffled when exhausted, so every sample is seen
    at a steady rate despite the ~1:9.5 class imbalance.
    """

    def __init__(
        self,
        dataset: CelebDFFaces,
        batch_size: int = 32,
        num_batches: int | None = None,
        seed: int = 42,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even to split evenly across 2 classes")
        missing = {0, 1} - set(dataset.indices_by_label)
        if missing:
            raise ValueError(f"dataset is missing class(es) {missing}; cannot balance")

        self.batch_size = batch_size
        self.per_class = batch_size // 2
        self.num_batches = num_batches
        self.rng = random.Random(seed)
        self._pools = {
            label: list(idx) for label, idx in dataset.indices_by_label.items()
        }
        self._cursors = {label: 0 for label in self._pools}
        for label in self._pools:
            self.rng.shuffle(self._pools[label])

    def _take(self, label: int, n: int) -> list[int]:
        out: list[int] = []
        pool, cur = self._pools[label], self._cursors[label]
        while len(out) < n:
            if cur >= len(pool):
                self.rng.shuffle(pool)
                cur = 0
            take = min(n - len(out), len(pool) - cur)
            out.extend(pool[cur : cur + take])
            cur += take
        self._cursors[label] = cur
        return out

    def __iter__(self):
        emitted = 0
        while self.num_batches is None or emitted < self.num_batches:
            batch = self._take(0, self.per_class) + self._take(1, self.per_class)
            self.rng.shuffle(batch)
            yield batch
            emitted += 1

    def __len__(self) -> int:
        if self.num_batches is None:
            raise TypeError("sampler is infinite; pass num_batches for a length")
        return self.num_batches


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation, for visualising crops in the report."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)
