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

---

# 10,000-iteration grid (2026-11-22)

Motivation: the 4000-iteration budget was a compute concession, and the paper
quotes 27k-40.5k iterations to convergence. This re-runs the paper's own grid at
10,000 iterations to answer the obvious objection -- "you undertrained".

Seed 42 throughout, `--select-by auc`, identical data and protocol to the 4k
runs. Run names carry `_s42_i10k`; results in `results10k/`.

## Test partition, frame level

| backbone | mode | protocol | acc | bal_acc | AUC | best iter |
|---|---|---|---|---|---|---|
| xception | none | identity | 0.9544 | 0.9503 | 0.9895 | 6000 |
| xception | hfc | identity | 0.9449 | 0.9530 | 0.9915 | 4000 |
| xception | hfc_only | identity | 0.9714 | 0.9430 | 0.9913 | 10000 |
| xception | none | random | 0.9842 | 0.9460 | 0.9951 | 9000 |
| xception | hfc | random | 0.9786 | 0.9613 | 0.9940 | 8000 |
| xception | hfc_only | random | 0.9822 | 0.9459 | 0.9943 | 5000 |
| efficientnet_b4 | none | identity | 0.9767 | 0.9541 | 0.9934 | 9000 |
| efficientnet_b4 | hfc | identity | 0.9783 | 0.9277 | 0.9915 | 9000 |

## Finding 1 -- the two protocols converge at different rates

| cell | d bal_acc (4k->10k) | d AUC |
|---|---|---|
| baseline / random | -0.0028 | +0.0004 |
| baseline / identity | **+0.0310** | **+0.0116** |

The random-split baseline was already converged at 4000 iterations; the
identity-split baseline was not, and gained another 0.0116 AUC. This is
mechanistically sensible: the random split leaks identities between train and
test, so a model can reach high scores early by memorising faces, while the
identity split forbids that shortcut and demands genuine forgery features, which
take longer to learn.

**Consequence: a fixed-budget random-vs-identity comparison overstates the
identity bias**, because it compares a converged run against an unconverged one.

| budget | identity - random, d bal_acc | d AUC |
|---|---|---|
| 4k | -0.0295 | -0.0168 |
| 10k | **+0.0043** | **-0.0056** |
| paper (Table II) | -0.0799 | -0.0282 |

The bias shrinks 3x on AUC and reverses sign on balanced accuracy. It does not
vanish -- AUC still drops -- but most of what we measured at 4k was a
convergence artifact, not leakage. The paper compares the two protocols at a
single fixed budget and so is exposed to the same confound.

## Finding 2 -- the paper's Table III ordering is reproduced at 10k

Our 4k runs found `hfc_only` best, reversing the paper's ablation. That does not
survive the longer budget.

Identity split, Xception: `hfc` 0.9530 bal_acc / 0.9915 AUC; `none` 0.9503 /
0.9895; `hfc_only` 0.9430 / 0.9913. So **hfc > none > hfc_only on balanced
accuracy, which is the paper's stated ordering.**

HFC effect (hfc - none), by budget:

| cell | 4k bal_acc | 4k AUC | 10k bal_acc | 10k AUC |
|---|---|---|---|---|
| xception / random | -0.0055 | -0.0011 | **+0.0153** | -0.0011 |
| xception / identity | -0.0030 | +0.0089 | **+0.0027** | +0.0020 |
| b4 / identity | -0.0344 | -0.0045 | -0.0264 | -0.0019 |

The balanced-accuracy sign flips from negative to positive on both Xception
protocols. The identity AUC gain shrinks from +0.0089 to +0.0020 -- closer to
the paper's own +0.0059 but on the same order.

EfficientNet-B4 remains the one cell that does not reproduce: HFC loses on both
metrics at both budgets.

## Finding 3 -- checkpoint selection is itself a noise source

Iteration chosen by best validation AUC, across the eight 10k runs: 4000, 5000,
6000, 8000, 9000, 9000, 9000, 10000. Spread 4000-10000, mean 7500, with no
pattern by mode or protocol. Validation loss swings widely between evaluation
points (e.g. 0.0655 -> 0.1717 -> 0.1930 over three consecutive evals of
`xception/none/identity`). The selected checkpoint is therefore partly a draw
from a noisy sequence, which is an additional variance term on top of seed
variance and is not reported in the paper.

Note also that `lr` remains 1.00e-03 in every logged line of every run:
ReduceLROnPlateau never fired within 10k iterations.

## The completed grid -- all four of the paper's cells at 10k (seed 42)

| cell | none bal_acc | hfc bal_acc | d bal | none AUC | hfc AUC | d AUC |
|---|---|---|---|---|---|---|
| xception/random | 0.9460 | 0.9613 | +0.0153 | 0.9951 | 0.9940 | -0.0011 |
| xception/identity | 0.9503 | 0.9530 | +0.0027 | 0.9895 | 0.9915 | +0.0020 |
| efficientnet_b4/random | 0.9583 | 0.9609 | +0.0026 | 0.9942 | 0.9969 | +0.0027 |
| efficientnet_b4/identity | 0.9541 | 0.9277 | -0.0264 | 0.9934 | 0.9915 | -0.0019 |

HFC wins on AUC in **2 of 4** cells and on balanced accuracy in 3 of 4. The mean
AUC effect across the four cells is **+0.0004**.

## Finding 4 -- the seed variance exceeds every effect being argued about

Three seeds (42, 43, 44) of `xception/{none,hfc_only}/identity` at 10k.

| config | metric | s42 | s43 | s44 | mean | sd | range |
|---|---|---|---|---|---|---|---|
| none | bal_acc | 0.9503 | 0.9533 | 0.8804 | 0.9280 | 0.0413 | **0.0729** |
| none | AUC | 0.9895 | 0.9923 | 0.9834 | 0.9884 | 0.0046 | **0.0089** |
| hfc_only | bal_acc | 0.9430 | 0.9108 | 0.9283 | 0.9274 | 0.0161 | 0.0322 |
| hfc_only | AUC | 0.9913 | 0.9917 | 0.9901 | 0.9910 | 0.0008 | 0.0016 |

The baseline's frame-AUC standard deviation across seeds is **0.0046**, with a
range of **0.0089** over only three runs. Set every claimed effect against it:

| claimed AUC effect | size | multiple of seed sd |
|---|---|---|
| paper, Xception/random (Table III) | +0.0012 | 0.26 sd |
| paper, Xception/identity (Table III) | +0.0059 | 1.30 sd |
| paper, EfficientNet-B4/identity (Table III) | +0.0026 | 0.57 sd |
| ours 10k, Xception/identity `hfc` | +0.0020 | 0.44 sd |
| ours 10k, Xception/identity `hfc_only` | +0.0018 | 0.40 sd |
| ours 4k, Xception/identity `hfc` | +0.0089 | 1.96 sd |

**Not one of them reaches 2 sd.** Averaged over three seeds, `hfc_only` beats
`none` by +0.0026 AUC (0.99 standard errors) and loses by 0.0006 balanced
accuracy (0.02 se). Both are consistent with no effect.

This is the central result of the reproduction. The paper reports single-run
numbers and its largest claimed gain is 1.3 standard deviations of the seed
noise we measured under its own protocol. The experiment as designed cannot
resolve the effect it is being used to demonstrate -- and neither can ours.
The honest reading is not "HFC does not work" but "this benchmark, at this
level of saturation, cannot tell."

## Finding 5 -- every high-pass front end is more stable than the baseline

A real difference does show up, just not in the mean. Seed standard deviation at
10k on Xception/identity, three seeds each, ranked:

| front end | frame AUC sd | bal_acc sd | AUC range | AUC sd vs baseline |
|---|---|---|---|---|
| `hfc_only` (high-pass only) | 0.0008 | 0.0161 | 0.0016 | **5.8x lower** |
| `learned` (constrained, N=4) | 0.0015 | 0.0174 | 0.0030 | 3.1x lower |
| `eq1` (the paper's filter) | 0.0017 | 0.0200 | 0.0034 | 2.7x lower |
| `pyramid` (Laplacian, L=4) | 0.0025 | 0.0300 | 0.0048 | 1.8x lower |
| `none` (no HFC) | 0.0046 | 0.0413 | 0.0089 | -- |

**All four high-pass variants are more repeatable than the baseline, on both
metrics, without exception.** This is the one claim in the whole study that
survives the noise floor, because it *is* a statement about the noise floor.
Forcing the first layer to read a DC-free representation appears to constrain
the optimisation toward a more consistent solution, even where it does not
raise the mean.

Ordering is suggestive: the fewer channels the front end admits and the harder
the high-pass constraint, the tighter the spread. `hfc_only` (one band, no
colour path) is tightest; `pyramid` (four bands) is loosest of the four.

With n=3 per configuration this is an observation, not a significance test, and
it is reported as such. But the pattern is 4 for 4 across two metrics.

# Front-end comparison at 10k, three seeds each (2026-11-22)

Xception, identity split, 10,000 iterations, `--select-by auc`, seeds 42/43/44.
Test partition, frame level. 8 new runs (~8 GPU-hours) on two accounts.

| front end | bal s42 | s43 | s44 | bal mean | sd | AUC s42 | s43 | s44 | AUC mean | sd |
|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 0.9503 | 0.9533 | 0.8804 | **0.9280** | 0.0413 | 0.9895 | 0.9923 | 0.9834 | **0.9884** | 0.0046 |
| `eq1` (paper) | 0.9530 | 0.9592 | 0.9219 | **0.9447** | 0.0200 | 0.9915 | 0.9931 | 0.9897 | **0.9914** | 0.0017 |
| `hfc_only` | 0.9430 | 0.9108 | 0.9283 | **0.9274** | 0.0161 | 0.9913 | 0.9917 | 0.9901 | **0.9910** | 0.0008 |
| `pyramid` | 0.9460 | 0.8930 | 0.9437 | **0.9276** | 0.0300 | 0.9921 | 0.9873 | 0.9888 | **0.9894** | 0.0025 |
| `learned` | 0.9565 | 0.9385 | 0.9217 | **0.9389** | 0.0174 | 0.9916 | 0.9897 | 0.9886 | **0.9900** | 0.0015 |

## Finding 6 -- neither extension beats the paper's hand-set filter

Against `eq1`, averaged over three seeds:

| front end | d bal_acc | d AUC | d AUC / eq1 seed sd | resolvable at n=3? |
|---|---|---|---|---|
| `pyramid` (multi-scale) | -0.0171 | -0.0020 | -1.20 sd | no |
| `learned` (constrained) | -0.0058 | -0.0015 | -0.86 sd | no |
| `hfc_only` (ablation) | -0.0173 | -0.0004 | -0.24 sd | no |
| `none` (no HFC) | -0.0167 | -0.0030 | -1.78 sd | no |

**An honest negative.** Both of our generalisations land slightly *below* the
single hand-set Gaussian the paper chose empirically, and both differences are
inside the seed noise. Multi-scale sampling and learned coefficients do not buy
accuracy on this benchmark.

Two readings, and the evidence does not separate them:

1. The paper's sigma=3 on a 3x3 grid is already near-optimal for this task, so
   there is nothing for the extra capacity to find.
2. The benchmark is saturated (every configuration sits at AUC 0.988-0.993), so
   no front-end change can show up regardless of merit.

Finding 4 argues for (2). Note that `pyramid` is also the *least* stable of the
four high-pass variants, which is weak evidence that four bands is extra
freedom the optimiser does not need here.

## Finding 7 -- the 4k single-seed ranking was pure noise

Compare the two rounds directly. 4,000 iterations, one seed, ranked by AUC:

| rank (4k, s42) | front end | bal_acc | AUC |
|---|---|---|---|
| 1 | `hfc_only` | 0.9382 | 0.9897 |
| 2 | `learned` | 0.9340 | 0.9882 |
| 3 | `pyramid` | 0.9276 | 0.9879 |
| 4 | `eq1` | 0.9163 | 0.9868 |
| 5 | `literal` | 0.9141 | 0.9857 |
| 6 | `none` | 0.9193 | 0.9779 |

10,000 iterations, three seeds, ranked by mean AUC:

| rank (10k, 3 seeds) | front end | bal_acc | AUC |
|---|---|---|---|
| 1 | `eq1` | 0.9447 | 0.9914 |
| 2 | `hfc_only` | 0.9274 | 0.9910 |
| 3 | `learned` | 0.9389 | 0.9900 |
| 4 | `pyramid` | 0.9276 | 0.9894 |
| 5 | `none` | 0.9280 | 0.9884 |

`eq1` moves from **last among the high-pass front ends to first**. At 4k it was
the only variant that lost to the no-HFC baseline on balanced accuracy, which
was the stated motivation for building the extensions at all. At 10k with three
seeds that result is gone: `eq1` leads on both metrics.

The 4k conclusion was not a small error, it was the wrong sign, and it was
produced by a single seed at a budget where training had not converged. This is
Finding 3 generalised: **the whole front-end ranking was noise.**

## Finding 8 -- the cherry-picking illusion, quantified

Best and worst seed per front end, frame AUC:

| front end | best | worst | spread |
|---|---|---|---|
| `none` | 0.9923 (s43) | 0.9834 (s44) | 0.0089 |
| `eq1` | 0.9931 (s43) | 0.9897 (s44) | 0.0034 |
| `pyramid` | 0.9921 (s42) | 0.9873 (s43) | 0.0048 |
| `learned` | 0.9916 (s42) | 0.9886 (s44) | 0.0030 |
| `hfc_only` | 0.9917 (s43) | 0.9901 (s44) | 0.0016 |

The baseline's **best** seed (0.9923) beats the three-seed **mean** of every
single HFC variant, including the paper's own. Run one seed of `none` and one of
`pyramid` and you can publish "HFC costs 0.0050 AUC"; swap which seeds you drew
and you can publish "HFC gains 0.0087". Both from the same code on the same
data. A single-run table on this benchmark carries almost no information about
the architecture, and that is the methodological point of the reproduction.

## Status -- experiments complete

22 runs at 10,000 iterations, roughly 20 GPU-hours across four Colab accounts:

- 4 paper cells (±HFC x {random, identity}) on Xception and EfficientNet-B4
- `hfc_only` ablation on both Xception protocols
- 3 seeds each for `none`, `eq1`, `hfc_only`, `pyramid`, `learned` on
  Xception/identity
- robustness sweep retained at 4,000 iterations (14 conditions x 2
  checkpoints), labelled as such

Not run: robustness at 10k (the two 10k identity checkpoints sit on an account
that hit its usage limit), `literal` at 10k, seeds on the random split or on B4.
