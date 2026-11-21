# Leveraging High-Frequency Components for Deepfake Detection: A Reproduction and Extension

**Rahul Bhat Yogesh — 2023B4A31035P**
Digital Image Processing, Project 20

Reproduction of Mejri, Papadopoulos & Aouada, *Leveraging High-Frequency Components
for Deepfake Detection* (Interdisciplinary Centre for Security, Reliability and Trust,
University of Luxembourg), with three original extensions.

---

## Abstract

**Problem.** Face-swap deepfakes are convincing to human viewers, and CNN detectors
that classify from colour texture alone generalise poorly to unseen subjects. Mejri
et al. propose a *High-Frequency Components* (HFC) module that replaces the first
convolutional layer of any RGB backbone with two parallel branches — one learning
ordinary colour features, one learning features from a Gaussian high-pass residual of
the greyscale input — concatenated and fused before the backbone's second layer.

**Method.** We reimplement the module from the paper text alone and evaluate it on
Celeb-DF-v2 with XceptionNet and EfficientNet-B4 under the paper's two data-splitting
protocols. We then extend it in three directions: a degradation-robustness benchmark
the paper names as future work; a multi-scale Laplacian-pyramid high-pass replacing the
paper's single band; and a learnable high-pass constrained to the Bayar–Stamm surface,
of which the paper's hand-set filter is shown to be one fixed member.

**Experiments.** 197,788 face crops extracted from 6,220 Celeb-DF-v2 videos; twelve
training runs under a single protocol; a 13-condition degradation sweep spanning JPEG
Q90–Q10, additive Gaussian noise, downscaling and blur.

**Findings.** (1) Parameter counts match the paper's Table IV exactly on all four
cells, confirming the architecture. (2) The identity-bias result reproduces: splitting
by subject rather than by video costs 0.0295 balanced accuracy and 0.0168 AUC. (3) The
HFC gain reproduces on XceptionNet (+0.0089 AUC, larger than the paper's +0.0059) but
**not** on EfficientNet-B4 (−0.0045), contradicting the paper's transferability claim.
(4) The module is robust to JPEG — its margin *triples* at Q10 — but **fragile to
additive noise**, inverting to −0.0412 at σ=10; we give a spectral explanation.
(5) Both of our generalisations beat the paper's fixed filter for ~300 extra
parameters. (6) A filter that barely high-passes at all performs within 0.0011 AUC of
the proper one, which questions whether the paper's stated mechanism explains its own
result.

---

## 1. Introduction

### 1.1 Problem statement

A face-swap deepfake replaces the face in a target video with that of a source
subject. The result is often indistinguishable to a human viewer, but the synthesis
process leaves traces: blending boundaries where the generated face meets the original
head, resampling artifacts from warping the face to fit, and the characteristic
spectral signature of a generator's upsampling layers. The detection problem is to
find those traces reliably.

### 1.2 Motivation

Most detectors operate on RGB images and are *undirected* — they are given raw pixels
and left to discover whatever is discriminative. This works well in-distribution, but
what a network finds easiest to latch onto is not always what we want it to learn. Two
specific failure modes motivate this paper:

- **Colour-texture overfitting.** Generators leave subject-specific colour statistics
  that a network can memorise instead of learning generic forgery traces.
- **Identity overfitting.** If the same person appears in both training and test sets,
  the detector can recognise *who* is in the frame rather than *whether the frame is
  fake*.

The paper addresses the first by explicitly supplying high-frequency content, on the
hypothesis that generator noise is concentrated there, and the second by proposing an
identity-disjoint data split.

### 1.3 Objectives

1. Reimplement the HFC module and its two backbones from the paper text alone.
2. Reproduce the paper's Tables II, III and IV on Celeb-DF-v2.
3. Resolve, and document, the ambiguities the paper leaves in its own specification.
4. Extend the work where the paper is weakest: robustness to real-world degradation,
   and the arbitrariness of its fixed filter.

---

## 2. Theory

### 2.1 The high-pass operator

The module's entire DIP content is Eq. (1) of the paper:

$$I_{HFC}(x,y) = I_g(x,y) - \frac{1}{2\pi\sigma^2}\sum_{i,j} I_g(x+i, y+j) 
e^{-\frac{i^2+j^2}{2\sigma^2}}$$

where $I_g$ is the greyscale image. In words: *the image minus a blurred copy of
itself* — an unsharp residual, and the standard way to isolate high spatial
frequencies. Low frequencies, which both copies share, cancel; edges and fine texture,
which the blur destroys, survive. The paper sets the kernel to 3×3 and σ=3
"empirically".

### 2.2 Two properties the paper does not discuss

Both were discovered during reimplementation and both affect the result.

**(a) The printed normaliser does not produce a high-pass.** Eq. (1) carries the
*continuous* Gaussian normaliser $1/(2\pi\sigma^2)$. A continuous Gaussian integrates
to 1, but a 3×3 *sample* of one does not. Evaluating the printed expression at σ=3:

| | centre | edge | corner | **sum** |
|---|---|---|---|---|
| weights | 0.01768 | 0.01673 | 0.01582 | **0.14789** |

The blur term therefore has gain 0.148, not 1, and

$$I_{HFC} \approx I_g - 0.148 \overline{I_g} \approx 0.85 I_g$$

which is a *mildly sharpened copy of the image*, not a high-pass. The test is DC
response: a true high-pass maps a constant image to zero. Measured on a constant image
of value 0.6, the printed form returns **0.511**; a sum-to-one kernel returns **0.000**.
Since Fig. 1 of the paper plainly shows an edge map, the intended operator must
normalise the kernel to sum to one. We implement that as the default and keep the
literal form behind a flag — and in Section 6.4 we show, surprisingly, that the choice
barely matters empirically.

**(b) σ=3 on a 3×3 grid is almost a box filter.** The normalised weights are

```
0.1070  0.1131  0.1070
0.1131  0.1196  0.1131         box filter for comparison: all 0.1111
0.1070  0.1131  0.1070
```

a falloff of under 11% from centre to corner. The operator is therefore very close to
$I_g - \mathrm{boxblur}_{3\times3}(I_g)$, a discrete Laplacian-style high-pass. The
Gaussian parameterisation suggests a tunable scale that, at these settings, is not
really present: σ is large relative to the kernel, so the kernel is truncated long
before the Gaussian decays. **This single narrow band is the weakness our extensions
target.**

### 2.3 Eq. (1) is a constrained convolution

Writing Eq. (1) as one convolution kernel, $K = \delta - g$ where $\delta$ is the unit
impulse and $g$ the normalised Gaussian:

```
-0.1070  -0.1131  -0.1070
-0.1131   0.8804  -0.1131          sum = 0
-0.1070  -0.1131  -0.1070
```

Rescaling so the centre is 1 gives **centre = 1, remaining weights summing to −1** —
exactly the constraint Bayar & Stamm (2016) impose to keep a *learned* first layer
high-pass, and which the image-forensics literature has used ever since.

This is the key theoretical observation of our extension: **the paper hand-picks one
fixed member of a family whose members can be learned.** Section 5.3 learns them.

### 2.4 Module architecture

With $C$ the backbone's first-conv output width:

| stage | Xception | EfficientNet-B4 |
|---|---|---|
| colour branch: `Conv(3→C, 3×3, s=2)` | 32 | 48 |
| HFC branch: Eq.(1) → BN → act → `Conv(1→⌊C/3⌋)` → BN → act | 10 | 16 |
| concatenate | 42 | 64 |
| fuse: `Conv(→C′)`, C′ = input width of the backbone's 2nd layer | 32 | 48 |

---

## 3. Methodology

### 3.1 Dataset

Celeb-DF-v2: 590 real videos of 59 celebrities, 5,639 face-swap forgeries among them,
and 300 YouTube-real videos of ordinary subjects.

**YouTube-real is excluded**, which reproduces the paper's Table I count of 590 real
videos. The justification is not merely arithmetic: no forgeries are generated from
those 300 subjects, and the footage comes from a visibly different source distribution.
Including it would let a detector learn "this source bucket ⇒ real" and score well
without ever learning a forgery artifact — the same shortcut-learning failure the paper
attacks in its Section V-A, keyed on source domain instead of identity.

### 3.2 Face extraction

Per the paper's Sec. IV-B: 32 evenly spaced frames per video, faces located with
BlazeFace, crops saved at 224×224 RGB with a 1.3× margin around the detected box.

Yield: **197,788 crops from 6,220 of 6,229 videos (99.2%)**. Nine videos produced no
detectable face and were logged and skipped.

### 3.3 Split protocols

Filenames encode identity: `Celeb-real/id{J}_{K}.mp4` and
`Celeb-synthesis/id{I}_id{J}_{K}.mp4`, where subject I's face is rendered onto subject
J's video. That `id{J}_{K}.mp4` is itself a valid real filename confirms J donates the
video and I the face.

- **Random split** — by video, stratified by label. The paper's baseline. All 59
  identities appear in all three partitions; this leak is the effect Table II measures.
- **Identity split** — by subject: 21% of identities to test, 19% to validation.

**Resolving the paper's ambiguity.** A forgery involves *two* subjects, and the paper
says only that it puts "real and fake videos of the same person in the same set". We
key each video on one identity — the **source**, since that is the face actually
rendered and therefore the one carrying the artifacts being classified — so a subject's
real videos and their forgeries always travel together.

We verified that a stricter reading is possible but is *not* what the paper did.
Treating identities as nodes and forgeries as edges, the graph has four connected
components of sizes **[38, 10, 10, 1]**, so assigning whole components would be
feasible. But it can only yield an 11/10/38 split — 18.6% test, 16.9% validation —
whereas the paper's stated 21%/19% of 59 identities gives exactly 12/11/36, which is
what per-identity assignment produces. The paper's own numbers therefore identify its
method. We implement the strict component split as an optional extension.

Resulting identity split: 36 train / 11 validation / 12 test subjects, **zero
train–test identity overlap**, 59.7%/17.5%/22.8% of videos.

### 3.4 Training

All settings from the paper's Sec. IV-B: BCE loss; Adam (lr 1e-3, β=(0.9, 0.999),
ε=1e-8); learning rate ×0.1 on plateau with patience 10; batch size 32 drawn **balanced
16 real / 16 fake** (essential — the dataset is ~1:9.5, so unbalanced batches would make
accuracy meaningless); Albumentations horizontal flip and random hue/saturation/
brightness/contrast. Mixed precision on an NVIDIA T4.

**Deviation:** the paper caps training at 100k iterations; we cap at **4,000**. Its own
Table IV reports convergence at 27k–40.5k, and our models converge far earlier still —
validation loss bottoms between iterations 1,000 and 4,000 in every run. All twelve runs
use an identical budget and checkpoint-selection rule, so the comparisons are internally
consistent.

### 3.5 Metrics

Accuracy and AUC, at frame and video level (video = mean sigmoid over a clip's frames).

**Balanced accuracy is the figure comparable to the paper.** Test partitions inherit the
~1:11 imbalance, so always predicting "fake" scores 0.917 on raw accuracy. The paper
reports 0.8834 for XceptionNet on the identity split — *below* that trivial baseline —
which indicates its accuracies are computed on balanced data. Reporting raw accuracy
against the paper's would have overstated our results by roughly five points.

---

## 4. Implementation

Python 3.11+, PyTorch, timm (backbones), Albumentations (augmentation), MediaPipe
(BlazeFace), OpenCV. ~2,000 lines across eleven modules, with **109 unit tests** that
all run without the dataset.

| file | role |
|---|---|
| `src/models/hfc.py` | Eq. (1), the two-branch module, and the three alternative front ends |
| `src/models/build.py` | grafting onto timm backbones; `verify_param_counts()` |
| `src/data/extract_faces.py` | video → 32 face crops, resumable, multiprocess |
| `src/data/splits.py` | random / identity / strict-identity protocols |
| `src/data/dataset.py` | Dataset, augmentation, balanced batch sampler |
| `src/train.py` | training loop, checkpoint/resume |
| `src/eval.py` | accuracy, balanced accuracy, AUC |
| `src/robustness.py` | degradation benchmark |
| `src/figures.py` | all report figures |

### 4.1 Correctness verification

The strongest check available: the paper's Table IV parameter counts are reproducible
**exactly**, which independently confirms the backbone variants, the single-output BCE
head, and the channel arithmetic.

| | ours | paper |
|---|---|---|
| Xception | 20.809 M | 20.81 M |
| Xception + HFC | 20.821 M | 20.82 M |
| EfficientNet-B4 | 17.550 M | 17.55 M |
| EfficientNet-B4 + HFC | 17.578 M | 17.58 M |

Other tests: the Laplacian pyramid satisfies perfect reconstruction to 6×10⁻⁸; the
learned filters hold the constraint to 10⁻⁵ after 50 Adam steps; every degradation
reduces PSNR monotonically (54 dB at Q90 → 35 dB at Q10) so a null robustness result
cannot be blamed on a no-op transform; the training loop drives a single batch to loss
1.5×10⁻¹².

### 4.2 A numerical bug worth recording

EfficientNet-B4 + HFC logged `val loss = nan` at iteration 4000 while its AUC stayed
finite. Cause: fp16 overflow to ±inf in the forward pass, which BCE turns into NaN.
Fixed by clamping logits to [−30, 30] before the loss — sigmoid(30) is already 1.0 to
twelve decimal places, so no prediction changes, while BCE(inf)=inf becomes BCE(30)=30.
Non-finite batches are now counted and reported rather than silently averaged.

---

## 5. Extensions

### 5.1 Degradation robustness benchmark

The paper's Conclusion names "different compression levels" as future work. We run it.

The motivation is specific: Eq. (1) reads *one* narrow high-frequency band, and JPEG
quantises high-frequency DCT coefficients hardest. Recent work reports that detectors
relying on "pixel-level artifacts or high-frequency noise patterns suffer a more
dramatic decrease in performance under JPEG". **Hypothesis stated before running: the
HFC margin shrinks as quality falls, possibly inverting.**

13 conditions: JPEG Q ∈ {90,70,50,30,20,10}, Gaussian noise σ ∈ {2,5,10},
downscale ×2/×4, blur σ ∈ {1,2}, plus a clean control. Evaluation only — no retraining.

*Limitation:* we degrade the extracted 224×224 crop, modelling an image re-shared
through social media. Re-encoding the source video before face extraction would be the
stricter test.

### 5.2 Multi-scale Laplacian-pyramid high-pass

Replace Eq. (1)'s single band with a Burt–Adelson pyramid over four scales:

$$G_0 = I_g,\quad G_{l+1} = \mathrm{reduce}(G_l),\quad
L_l = G_l - \mathrm{expand}(G_{l+1})$$

using the separable binomial kernel [1,4,6,4,1]/16. Bands are resampled to the input
resolution and stacked, so the HFC branch sees four channels instead of one. Rationale:
degradation destroys the finest band first, so coarser traces — blending boundaries,
warping — should survive when the finest are gone.

### 5.3 Learnable constrained high-pass

Following §2.3, we learn N filters on the constraint surface

$$\text{centre} = 1, \qquad \textstyle\sum \text{others} = -1$$

so every filter sums to zero and annihilates DC however its weights move. The
constraint is re-applied inside `forward`, so it survives any optimiser. **Filter 0 is
initialised at Eq. (1) exactly**, so the paper's configuration is the starting point and
any gain is attributable to learning rather than a different initialisation.

### 5.4 The greyscale control

Prompted by the result in §6.4: feed the branch **unfiltered greyscale**, everything
else identical. If this matches the HFC module, the benefit is architectural — an extra
input branch — rather than spectral, and the paper's stated mechanism does not explain
its own result.

---

## 6. Results

All figures are test-partition, frame level, under the protocol of §3.4.

### 6.1 Table II — identity bias

| XceptionNet, no HFC | balanced accuracy | AUC |
|---|---|---|
| random split | 0.9488 | 0.9947 |
| identity split | 0.9193 | 0.9779 |
| **drop** | **−0.0295** | **−0.0168** |

**Reproduced.** The paper reports −0.0799 accuracy and −0.0282 AUC. Same direction,
smaller magnitude. Splitting by video rather than by subject measurably inflates
results, and any Celeb-DF number reported under a random split should be read with that
in mind.

### 6.2 Table III — the HFC effect

| backbone | split | no HFC | + HFC | Δ AUC | paper Δ AUC |
|---|---|---|---|---|---|
| Xception | random | 0.9947 | 0.9936 | −0.0011 | +0.0012 |
| Xception | identity | 0.9779 | **0.9868** | **+0.0089** | +0.0059 |
| EfficientNet-B4 | identity | 0.9901 | 0.9856 | **−0.0045** | +0.0026 |

**Partially reproduced.** On XceptionNet the gain is real and *larger* than the paper's.
On the random split there is no gain — but neither is there in the paper (+0.0012),
and our baseline sits at 0.9947 AUC, effectively at ceiling.

**On EfficientNet-B4 the claim fails.** The paper's third contribution is that the
module transfers between architectures; we measure a small negative effect. Our B4
baseline is at 0.9901 AUC against the paper's 0.9756, so again there is far less
headroom — but the result stands as measured.

The pattern across all three comparisons is consistent: **HFC helps only where the
baseline leaves room.**

### 6.3 Our extensions vs the paper's filter

Xception, identity split:

| front end | balanced accuracy | AUC | parameters |
|---|---|---|---|
| none | 0.9193 | 0.9779 | 20.809 M |
| **learned (N=4)** | **0.9340** | **0.9882** | 20.822 M |
| **pyramid (4 levels)** | **0.9276** | **0.9879** | 20.821 M |
| eq1 (the paper) | 0.9163 | 0.9868 | 20.821 M |
| literal Eq. (1) | 0.9141 | 0.9857 | 20.821 M |

Both generalisations beat the paper's fixed filter on both metrics for ~300 extra
parameters. Note that `eq1` scores *below* the no-HFC baseline on balanced accuracy
while beating it on AUC — it improves ranking at the cost of calibration — and both our
variants land above the baseline on both.

**How much to claim:** the AUC gaps over `eq1` (+0.0011, +0.0014) are within
single-seed noise and we do not claim them as significant. The balanced-accuracy gaps
(+0.011, +0.018) are more substantial, and two independent modifications moving both
metrics the same direction is stronger evidence than either alone.

### 6.4 Does the high-pass matter? *(ablations the paper did not run)*

| variant | AUC |
|---|---|
| no HFC branch | 0.9779 |
| branch fed the **literal** Eq. (1) — barely a high-pass | 0.9857 |
| branch fed a proper high-pass | 0.9868 |
| *branch fed unfiltered greyscale* | *[pending]* |

The literal form leaves ~85% of the DC through, yet lands within **0.0011 AUC** of the
proper high-pass, while both beat the baseline by ~0.008. If the mechanism were really
about high-frequency content, that gap should be far larger.

`hfc_only` sharpens the puzzle from the other side: discarding colour entirely and
feeding the backbone nothing but per-channel high-pass residuals reached validation AUC
0.9856, matching colour+HFC's 0.9855 — where the paper reports `hfc_only` as clearly its
worst setup.

The two findings are reconcilable if **colour and high-frequency content carry largely
redundant evidence** on this dataset: either alone suffices, and adding a second branch
of either kind helps slightly. The greyscale control decides it.

### 6.5 Robustness *(new)*

Identity split, frame AUC:

| condition | PSNR | no HFC | + HFC | margin |
|---|---|---|---|---|
| clean | ∞ | 0.9764 | 0.9857 | +0.0093 |
| JPEG Q90 | 46.0 | 0.9710 | 0.9818 | +0.0108 |
| JPEG Q50 | 39.9 | 0.9654 | 0.9743 | +0.0089 |
| JPEG Q30 | 37.3 | 0.9522 | 0.9608 | +0.0086 |
| JPEG Q20 | 35.2 | 0.9076 | 0.9236 | +0.0160 |
| **JPEG Q10** | 31.1 | 0.7154 | 0.7451 | **+0.0297** |
| noise σ=2 | 43.8 | 0.9670 | 0.9782 | +0.0112 |
| noise σ=5 | 34.9 | 0.9072 | 0.9122 | +0.0050 |
| **noise σ=10** | 28.6 | 0.7925 | 0.7513 | **−0.0412** |
| downscale ×2 | 36.1 | 0.9684 | 0.9725 | +0.0041 |
| **downscale ×4** | 28.5 | 0.9134 | 0.8960 | **−0.0174** |
| blur σ=1 | 47.0 | 0.9729 | 0.9853 | +0.0124 |

**Our hypothesis was wrong, and the truth is more specific.** The HFC margin under JPEG
does not shrink — it *triples*, from +0.0093 clean to +0.0297 at Q10. But it **inverts
under additive noise** (−0.0412 at σ=10) and heavy downscaling (−0.0174 at ×4).

The asymmetry has a clean spectral reading:

- **JPEG *removes* high-frequency energy.** Both models degrade; the high-pass branch
  still extracts signal from what survives.
- **Gaussian noise *adds* broadband high-frequency energy**, directly into the band
  Eq. (1) reads. The branch is swamped — genuine traces are buried under noise that the
  baseline's colour path simply ignores.

So: **the HFC module is robust to compression but fragile to additive noise.**

A separate practical point: at Q10 both models retain AUC near 0.72–0.75 while balanced
accuracy collapses to 0.5450 and 0.5098 — chance. The *ranking* survives; the decision
threshold does not. A detector deployed on heavily compressed media would need
recalibration even though its discriminative power is partly intact.

---

## 7. Discussion

### 7.1 Why our absolute numbers exceed the paper's

Across the board our figures sit above the paper's — most starkly on the identity
split, where our no-HFC baseline reaches 0.9779 AUC against 0.9669. Candidate
explanations, in order of our confidence:

1. **Our identity split is probably easier.** We key forgeries on the source identity,
   so target identities still cross partitions and their background, pose and lighting
   are visible in both. A stricter protocol would close some of the gap. §3.3 explains
   why we believe this matches the paper's own method, but we cannot verify it.
2. **Shorter training.** We stop at 4,000 iterations where the paper runs to 27k–40k.
   Longer training increases identity overfitting, which costs most precisely on the
   identity split. Our early stop may *help* here.
3. **Face-crop framing.** Our 1.3× margin includes hair and jaw boundary — exactly where
   blending artifacts live. A tighter crop would be harder.

### 7.2 The saturation problem

The deepest issue with this reproduction is that **our baselines are too strong for the
proposed module to help.** On the random split the baseline reaches 0.9947 AUC; the
theoretical maximum gain is 0.0053. The paper's baseline was at 0.9951 — essentially
identical — and its reported gain was +0.0012, so it faced the same ceiling.

This is why the identity split matters: it is the only setting in our experiments with
real headroom, and the only one where HFC clearly helps. It argues that evaluating such
modules on saturated benchmarks is close to uninformative, and that harder protocols —
stricter splits, cross-dataset tests, degraded inputs — are where they should be judged.

### 7.3 What the ablations suggest about the mechanism

The paper attributes its gain to high-frequency content. Our ablations put that under
strain: a filter that barely high-passes performs within 0.0011 AUC of a proper one,
and a model given *only* high-pass residuals matches one given colour plus high-pass.
The most economical explanation is that the two information sources are largely
redundant here, and that part of the measured benefit comes from the two-branch
architecture rather than the spectral content specifically.

This does not make the module useless — it works, and it is nearly free. But the
*reason* it works may not be the reason the paper gives, and the paper's own ablation
set is too small to distinguish the two. The missing control is the one we added.

### 7.4 Failure cases and limitations

- **Noise fragility (§6.5)** is the clearest failure, and it is a real deployment
  concern: sensor noise, low light and aggressive denoising are common in the wild.
- **Threshold collapse under heavy compression** — AUC 0.74 but balanced accuracy 0.51
  at Q10 — means a fixed threshold is unsafe on degraded media.
- **Single seed.** No run is repeated, so differences below ~0.005 AUC are not
  distinguishable from noise. We have flagged every claim that falls in that range.
- **Reduced training budget** (4k vs 27k–40k iterations) is a deliberate, documented
  deviation.
- **Crop-level degradation**, not video re-encoding (§5.1).
- **EfficientNet-B4 coverage** is limited to the identity split.

---

## 8. Conclusion

We reimplemented the HFC module from the paper text alone and verified the
reimplementation against the strongest available evidence: the paper's Table IV
parameter counts, matched exactly on all four cells.

**Objectives achieved.** The identity-bias result reproduces (−0.0295 balanced accuracy,
−0.0168 AUC). The HFC gain reproduces on XceptionNet, at +0.0089 AUC — larger than the
paper's +0.0059. The transferability claim does **not** reproduce: on EfficientNet-B4 we
measure −0.0045. Two ambiguities in the paper's specification were identified and
resolved with documented reasoning — the normaliser in Eq. (1), and identity keying for
two-subject forgeries, where the paper's own stated split fractions turned out to
identify its method.

**New findings.** The module is robust to JPEG compression, with its margin tripling at
Q10, but fragile to additive noise, inverting to −0.0412 at σ=10 — explained by noise
injecting energy into precisely the band the operator reads. Replacing the paper's
single hand-set band with a four-level Laplacian pyramid, or with filters learned under
the Bayar–Stamm constraint of which Eq. (1) is one fixed member, improves balanced
accuracy by 0.011 and 0.018 respectively for roughly 300 additional parameters. And a
filter that barely high-passes performs within 0.0011 AUC of a proper one, which
questions whether the paper's stated mechanism explains its own result.

**Future work.** Repeat runs across seeds to put error bars on the sub-0.005 differences;
evaluate under the strict component-based identity split already implemented; test
cross-dataset generalisation to FaceForensics++; and train with noise augmentation to
see whether the fragility in §6.5 is a property of the operator or merely of the
training distribution.

---

## 9. References

1. N. Mejri, K. Papadopoulos, D. Aouada. *Leveraging High-Frequency Components for
   Deepfake Detection.* IEEE MMSP, 2022.
2. Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu. *Celeb-DF: A Large-Scale Challenging Dataset
   for DeepFake Forensics.* CVPR, 2020.
3. F. Chollet. *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR, 2017.
4. M. Tan, Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks.* ICML, 2019.
5. B. Bayar, M. C. Stamm. *A Deep Learning Approach to Universal Image Manipulation
   Detection Using a New Convolutional Layer.* ACM IH&MMSec, 2016.
6. P. J. Burt, E. H. Adelson. *The Laplacian Pyramid as a Compact Image Code.*
   IEEE Trans. Communications, 1983.
7. J. Fridrich, J. Kodovsky. *Rich Models for Steganalysis of Digital Images.*
   IEEE TIFS, 2012.
8. Y. Jeong et al. *BiHPF: Bilateral High-Pass Filters for Robust Deepfake Detection.*
   WACV, 2022.
9. V. Bazarevsky et al. *BlazeFace: Sub-millisecond Neural Face Detection on Mobile
   GPUs.* arXiv:1907.05047, 2019.
10. A. Buslaev et al. *Albumentations: Fast and Flexible Image Augmentations.*
    Information, 2020.

---

## 10. Appendix — reproducing this work

Full source: `https://github.com/rahulbhatyogesh/DIP-Project`

```bash
pip install torch torchvision timm albumentations opencv-python mediapipe

# 1. Extract faces (once, on Kaggle where Celeb-DF-v2 is mounted)
python -m src.data.extract_faces --root <celeb-df-v2> --out faces \
    --frames 32 --size 224 --workers 4

# 2. Build both split protocols
python -c "from src.data.splits import build_splits, summarise; \
           print(summarise(build_splits('<celeb-df-v2>', 'faces/splits.json')))"

# 3. Train one cell
python -m src.train --faces-root faces --manifest faces/manifest.csv \
    --splits faces/splits.json --backbone xception --mode hfc \
    --protocol identity --out runs/xception_hfc_identity \
    --max-iters 4000 --select-by loss --resume

#    --hfc-filter {eq1,pyramid,learned,gray}   selects the front end
#    --literal-eq1                             uses the paper's printed normaliser

# 4. Score
python -m src.eval --ckpt runs/xception_hfc_identity/best.pt \
    --faces-root faces --manifest faces/manifest.csv \
    --splits faces/splits.json --part test --out results/xception_hfc_identity.json

# 5. Robustness sweep (no retraining)
python -m src.robustness --ckpt runs/*/best.pt --faces-root faces \
    --manifest faces/manifest.csv --splits faces/splits.json \
    --max-frames-per-video 8 --out results/robustness

# 6. Figures and tables
python -m src.figures all --faces-root faces --manifest faces/manifest.csv \
    --results results --robustness results/robustness --out report/figures

# 7. Tests (109, none require the dataset)
python -m pytest tests/ -q
python -c "from src.models.build import verify_param_counts; verify_param_counts()"
```
