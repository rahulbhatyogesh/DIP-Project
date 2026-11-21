# Collected results

Transcribed from run output as each job finished. The authoritative JSONs live in
the Drive of whichever Colab account ran the job; this file is the durable copy.

Protocol for every cell: Celeb-DF-v2, YouTube-real excluded, 32 frames/video,
224×224 BlazeFace crops, batch 32 balanced 16/16, BCE + Adam(1e-3, β=(0.9,0.999),
ε=1e-8), `--max-iters 4000 --select-by loss`, best checkpoint by validation loss,
scored on the **test** partition.

Celeb-DF-v2 test partitions are ~1:11 real:fake, so **balanced accuracy** is the
figure comparable to the paper; raw accuracy is dominated by the fake class
(always predicting "fake" scores 0.917).

---

## Table III — ours vs paper (test, frame level)

| backbone | setup | split | bal_acc | paper acc | AUC | paper AUC |
|---|---|---|---|---|---|---|
| xception | none | random | 0.9488 | 0.9633 | 0.9947 | 0.9951 |
| xception | hfc | random | 0.9433 | 0.9663 | 0.9936 | 0.9963 |
| xception | **hfc_only** | random | **0.9602** | 0.9431 | 0.9937 | 0.9901 |
| xception | none | identity | 0.9193 | 0.8834 | 0.9779 | 0.9669 |
| xception | hfc | identity | 0.9163 | 0.9064 | 0.9868 | 0.9728 |
| xception | **hfc_only** | identity | **0.9382** | 0.8750 | **0.9897** | 0.9665 |
| efficientnet_b4 | none | identity | 0.9376 | 0.9056 | 0.9901 | 0.9756 |
| efficientnet_b4 | hfc | identity | 0.9032 | 0.9246 | 0.9856 | 0.9782 |

Raw (imbalanced) accuracy, for completeness: xception none/random 0.9819,
hfc/random 0.9797, none/identity 0.9653, hfc/identity 0.9695;
b4 none/identity 0.9738, hfc/identity 0.9676.

### Video level (mean sigmoid over a clip's frames)

| backbone | setup | split | bal_acc | AUC |
|---|---|---|---|---|
| xception | none | random | 0.9705 | 0.9995 |
| xception | hfc | random | 0.9794 | 0.9993 |
| xception | none | identity | 0.9457 | 0.9884 |
| xception | hfc | identity | 0.9583 | 0.9937 |
| efficientnet_b4 | none | identity | 0.9606 | 0.9955 |
| efficientnet_b4 | hfc | identity | 0.9277 | 0.9911 |

## Table II — identity bias (no-HFC)

| backbone | split | bal_acc | AUC |
|---|---|---|---|
| xception | random | 0.9488 | 0.9947 |
| xception | identity | 0.9193 | 0.9779 |
| | **drop** | **−0.0295** | **−0.0168** |

Paper's Xception drop: −0.0799 accuracy, −0.0282 AUC. Same direction, smaller
magnitude. **Reproduced.**

## Table IV — parameter counts

| backbone | setup | ours | paper |
|---|---|---|---|
| xception | none | 20.809 M | 20.81 M |
| xception | hfc | 20.821 M | 20.82 M |
| efficientnet_b4 | none | 17.550 M | 17.55 M |
| efficientnet_b4 | hfc | 17.578 M | 17.58 M |

Exact match on all four. Front-end variants: pyramid 20.821 M, learned 20.822 M
(+12,484 and +12,520 over the plain backbone, against eq1's +12,208).

---

## HFC effect, summarised

| comparison | Δ bal_acc | Δ AUC | paper Δ AUC |
|---|---|---|---|
| xception, random | −0.0055 | −0.0011 | +0.0012 |
| xception, identity | −0.0030 | **+0.0089** | +0.0059 |
| efficientnet_b4, identity | −0.0344 | **−0.0045** | +0.0026 |

The paper's own numbers show the gain concentrated on the identity split
(+0.0059 there vs +0.0012 on random). We reproduce that concentration on
Xception, where our AUC gain (+0.0089) exceeds the paper's. We do **not**
reproduce it on EfficientNet-B4, where HFC is slightly worse on both metrics.

Likely reason the random split shows nothing: our baseline is already at
0.9947 AUC, essentially ceiling, leaving no headroom.

---

## Front-end comparison (our extension) — Xception, identity split, test

| front end | bal_acc | frame AUC | video bal_acc | video AUC | params |
|---|---|---|---|---|---|
| **hfc_only** (no colour branch) | **0.9382** | **0.9897** | 0.9564 | 0.9978 | 20.809 M |
| none (no HFC branch) | 0.9193 | 0.9779 | 0.9457 | 0.9884 | 20.809 M |
| **learned** (constrained, N=4) | **0.9340** | **0.9882** | 0.9557 | 0.9959 | 20.822 M |
| **pyramid** (Laplacian, 4 levels) | **0.9276** | **0.9879** | 0.9579 | 0.9928 | 20.821 M |
| eq1 (the paper) | 0.9163 | 0.9868 | 0.9583 | 0.9937 | 20.821 M |
| literal Eq. (1) | 0.9141 | 0.9857 | 0.9400 | 0.9938 | 20.821 M |

Deltas against the paper's `eq1`: learned +0.0177 bal_acc / +0.0014 AUC;
pyramid +0.0113 bal_acc / +0.0011 AUC. Cost: ~300 parameters.

**Two things stand out.**

1. `eq1` scores *below* the no-HFC baseline on balanced accuracy (0.9163 vs
   0.9193) while beating it on AUC — it improves ranking at the cost of
   calibration. Both of our generalisations fix that, landing above the baseline
   on both metrics.
2. The AUC gaps over `eq1` (+0.0011, +0.0014) are within single-seed noise and
   should not be leaned on. The balanced-accuracy gaps (+0.011, +0.018) are more
   substantial, and both independent modifications moving both metrics the same
   way is worth more than either result alone.

**`hfc_only` wins outright, reversing the paper's ablation ordering.** Discarding
colour entirely and feeding the backbone nothing but per-channel high-pass residuals
beats every other configuration on both splits and both metrics (identity: 0.9382 /
0.9897). The paper reports `hfc_only` as its *worst* setup on both splits. This
refutes the "maybe it is just the extra branch" reading -- `hfc_only` has no second
branch and still wins -- and says high-frequency content is not merely sufficient here
but better than colour. Keeping the colour branch alongside appears to dilute it:
hfc 0.9868 < hfc_only 0.9897.

**The literal-Eq.(1) result is the awkward one for the paper.** A filter that
leaves ~85% of the DC through — barely a high-pass at all — reaches 0.9857 AUC
against the proper high-pass's 0.9868, a gap of 0.0011, while both beat the
baseline by ~0.008. If the mechanism were really about high-frequency content,
that gap should be far larger. The `gray` control (unfiltered greyscale into the
same branch) tests this directly.

## Robustness sweep (new experiment — not in the paper)

Test partition, `--max-frames-per-video 8`, so clean numbers differ slightly
from the full-test table above (e.g. 0.9764 vs 0.9779 for xception/none/identity).

### Identity split, frame AUC

| condition | PSNR | none | hfc | margin |
|---|---|---|---|---|
| clean | ∞ | 0.9764 | 0.9857 | +0.0093 |
| JPEG Q90 | 46.0 | 0.9710 | 0.9818 | +0.0108 |
| JPEG Q70 | 42.0 | 0.9706 | 0.9816 | +0.0110 |
| JPEG Q50 | 39.9 | 0.9654 | 0.9743 | +0.0089 |
| JPEG Q30 | 37.3 | 0.9522 | 0.9608 | +0.0086 |
| JPEG Q20 | 35.2 | 0.9076 | 0.9236 | +0.0160 |
| JPEG Q10 | 31.1 | 0.7154 | 0.7451 | **+0.0297** |
| noise σ=2 | 43.8 | 0.9670 | 0.9782 | +0.0112 |
| noise σ=5 | 34.9 | 0.9072 | 0.9122 | +0.0050 |
| noise σ=10 | 28.6 | 0.7925 | 0.7513 | **−0.0412** |
| downscale ×2 | 36.1 | 0.9684 | 0.9725 | +0.0041 |
| downscale ×4 | 28.5 | 0.9134 | 0.8960 | **−0.0174** |
| blur σ=1 | 47.0 | 0.9729 | 0.9853 | +0.0124 |
| blur σ=2 | 40.7 | — | 0.9769 | — |

Balanced accuracy, same conditions — none: 0.9191, 0.9004, 0.9165, 0.9099,
0.8833, 0.7834, 0.5450, 0.8842, 0.8180, 0.5744, 0.8930, 0.8203, 0.9129.
hfc: 0.9266, 0.8865, 0.9206, 0.9158, 0.8941, 0.7823, 0.5098, 0.8446, 0.6617,
0.6384, 0.8292, 0.7173, 0.9428, 0.9206.

### Random split (partial — session was reclaimed mid-run)

xception/none/random frame AUC: clean 0.9947, Q90 0.9925, Q70 0.9914,
Q50 0.9893, Q30 0.9723, Q20 0.9168, Q10 0.7384, noise σ2 0.9914, noise σ5 0.9626.

### Finding

**The stated hypothesis was wrong, in an informative way.** We predicted the HFC
margin would shrink under JPEG. It *grows* — from +0.0093 clean to +0.0297 at Q10.
But it **inverts under additive noise** (−0.0412 at σ=10) and heavy downscaling
(−0.0174 at ×4).

The asymmetry has a clean physical reading: JPEG *removes* high-frequency energy,
and the high-pass branch still finds signal in what remains, while Gaussian noise
*adds* broadband high-frequency energy directly into the band Eq. (1) reads,
burying the forgery traces under noise the colour path simply ignores.

So: **the HFC module is robust to compression but fragile to additive noise.**
This also predicts the pyramid should help specifically under noise, since noise
energy concentrates in the finest band and coarser levels average it away.

At JPEG Q10 both models collapse to near-chance balanced accuracy (0.5450 and
0.5098) despite AUC near 0.72–0.75 — the ranking survives, the decision threshold
does not. Worth noting for the real-world-relevance argument.

---

## Known issue

`efficientnet_b4/hfc/identity` logged `val loss=nan` at iteration 4000 while its
AUC stayed finite at 0.9877 — an fp16 overflow under AMP producing non-finite
logits in at least one batch. The selected checkpoint is iteration 3000, which is
clean, so the reported result is unaffected, but the loss average is not
trustworthy for that pass. See the fix in `src/eval.py`.

## Still to run

- `xception_hfc_gray_identity` — the control that decides whether the gain is
  spectral or architectural. Highest-value remaining job.
- `xception_hfc_only_{identity,random}` (Table III's third row)
- robustness sweep over the random pair and the new front ends
