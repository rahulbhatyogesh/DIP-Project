# Leveraging High-Frequency Components for Deepfake Detection — reproduction

Implementation of Mejri, Papadopoulos & Aouada (SnT, University of Luxembourg)
on Celeb-DF-v2, for the DIP course project.

The paper's contribution is the **HFC module**: it replaces the first
convolutional layer of any RGB CNN with two parallel branches — ordinary colour
features, and features learned from a Gaussian high-pass residual of the
grayscale input — concatenated and fused before the backbone's second layer.

## Layout

```
src/
  config.py              every hyperparameter the paper specifies
  models/hfc.py          Eq. (1) + the two-branch module        <- core contribution
  models/build.py        grafting onto timm backbones
  data/extract_faces.py  video -> 32 face crops per clip
  data/splits.py         random vs identity split protocols
  data/dataset.py        Dataset, augmentation, balanced sampler
  train.py               BCE + Adam + plateau LR + checkpoint/resume
  eval.py                accuracy + AUC, frame and video level
tests/                   69 tests, all runnable without the dataset
```

## Install

```bash
pip install torch torchvision timm albumentations opencv-python mediapipe
```

`mediapipe` supplies BlazeFace, the detector the paper uses. If it will not
install, the extractor falls back to `facenet-pytorch`'s MTCNN via
`--detector mtcnn`.

## 1. Extract faces (once, on Kaggle)

Celeb-DF-v2 is already mounted in a Kaggle notebook, so there is nothing to
download. Stored as JPEGs the full crop set is only ~4–5 GB.

```bash
python -m src.data.extract_faces \
    --root /kaggle/input/celeb-df-v2 \
    --out  /kaggle/working/faces \
    --frames 32 --size 224 --workers 4
```

Then publish `/kaggle/working/faces` as a **private Kaggle dataset** and pull it
into any Colab account in a few minutes. Do not put it on Google Drive — the
free tier is 15 GB per account and is very slow with many small files.

Re-running skips videos already done, so a reclaimed session costs nothing.

## 2. Build the splits

```bash
python -c "from src.data.splits import build_splits, summarise; \
           print(summarise(build_splits('/kaggle/input/celeb-df-v2', 'splits.json')))"
```

Produces both protocols in one manifest:

- **random** — split by video. The paper's baseline, and it leaks every identity
  across partitions. That leak is the point: it is what inflates Table II.
- **identity** — split by subject, 21% of identities to test and 19% to val.
  Each video follows exactly one key identity, so a person's real videos and
  their fakes always travel together. The key is the **source** identity (the
  face actually rendered); `identity_key="target"` gives the alternative.

`YouTube-real` is excluded by default. No fakes are ever generated from those
300 subjects, so including them lets a model learn "this source bucket ⇒ real"
without learning any forgery artifact — and excluding them reproduces Table I's
count of 590 real videos.

## 3. Train (Colab)

```bash
python -m src.train \
    --faces-root faces --manifest faces/manifest.csv --splits splits.json \
    --backbone xception --mode hfc --protocol identity \
    --out runs/xception_hfc_identity --resume
```

- `--backbone` : `xception`, `efficientnet_b4`, `efficientnet_b0`
- `--mode`     : `none` (baseline) | `hfc` | `hfc_only` — the three rows of Table III
- `--protocol` : `random` | `identity` — the two columns of Table II

`--max-iters` defaults to the relevant Table IV convergence figure plus 25%
headroom (the paper's 100k is only a ceiling). Always pass `--resume`: free
Colab reclaims idle sessions, and checkpoints are written atomically every 1000
iterations so a kill is never destructive.

### Run schedule

| Tier | Runs | Gives you |
|---|---|---|
| 1 | Xception × {none, hfc} × {random, identity} | All of Table II + the paper's main claim |
| 2 | + EfficientNet-B4, same four cells | Transferability (Table III main rows) |
| 3 | + `hfc_only`, both backbones, both splits | Complete Table III |

Free Colab allows roughly 3–4 GPU-hours per account per day, so tier 2 wants
3–4 accounts in parallel.

## 4. Evaluate

```bash
python -m src.eval --ckpt runs/xception_hfc_identity/best.pt \
    --faces-root faces --manifest faces/manifest.csv --splits splits.json \
    --out results/xception_hfc_identity.json
```

Reports accuracy and AUC at both frame and video level (video = mean sigmoid
over a clip's 32 frames). The paper does not say which it uses, so both are
recorded.

## Verification

```bash
python -m pytest tests/ -q                 # 69 tests
python -c "from src.models.build import verify_param_counts; verify_param_counts()"
```

The parameter check is the strongest correctness signal available: the paper's
Table IV counts are reproducible exactly, which is what pins the architecture
down.

```
xception        none   ours= 20.809M  paper= 20.81M
xception        hfc    ours= 20.821M  paper= 20.82M
efficientnet_b4 none   ours= 17.550M  paper= 17.55M
efficientnet_b4 hfc    ours= 17.578M  paper= 17.58M
```

Before committing GPU hours, confirm the loop can memorise a single batch:

```bash
python -m src.train ... --overfit-batches 1 --no-pretrained
```

## Two decisions the paper leaves ambiguous

**1. Eq. (1) normalisation.** The equation carries the continuous Gaussian
normaliser `1/(2πσ²)`. Applied literally to a 3×3 kernel at σ=3 that factor sums
to **0.1479**, so the result is `I_HFC ≈ 0.85·I_g` — a mildly sharpened image,
not a high-pass. A constant image maps to 0.511 rather than 0. Fig. 1 of the
paper plainly shows an edge map, so the intended operator normalises the kernel
to sum to 1, which is the default here. `--literal-eq1` selects the printed
version for comparison.

**2. σ=3 truncated to 3×3 is almost a box filter.** The weights come out 1.000
(centre), 0.946 (edge), 0.895 (corner) — about 10% falloff across the whole
kernel, against a box filter's perfectly uniform 1/9. So the operator is very
close to `I_g − boxblur₃ₓ₃(I_g)`, a discrete Laplacian-style high-pass. Worth
stating in the report: it ties the method back to standard DIP sharpening.

## Note on early validation numbers

The first few validation passes often look terrible, sometimes with an enormous
loss, even when training is working. BatchNorm running statistics need a few
hundred iterations to converge, and until they do, eval-mode outputs are poorly
calibrated. Judge a run by its trend.
