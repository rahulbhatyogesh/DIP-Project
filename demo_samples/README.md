# Demo samples — files this detector gets right

100 images (`images/fake1..50.jpg`, `images/real1..50.jpg`) and 30 videos
(`videos/fake1..15.mp4`, `videos/real1..15.mp4`). `fake*` are Celeb-DF-v2
synthesis outputs; `real*` are the unmanipulated Celeb-real clips.

Within each class the numbering runs from most to least confident, so `fake1` /
`real1` are the safest files to open a demo with.

## How they were chosen

* **Model**: `runs/xception_pyramid_identity_s42_i10k/best.pt` — Xception with
  the HFC module, `pyramid` front end, identity split.
* **Source**: the **test partition of the identity protocol** only. The model
  never saw these clips *or these subjects* in training, so a correct verdict
  here is not memorisation. 79 distinct clips across 12 identities; the clips
  behind the images and the clips behind the videos are disjoint sets.
* **Pipeline**: exactly `src/app.py` — BlazeFace detection, 1.3x square margin
  crop, 224x224, ImageNet normalisation. Video verdict = mean sigmoid over the
  32 evenly spaced sampled frames, the same aggregation `src/eval.py` uses.
* **Acceptance**: every file is scored *after* being written, so the number in
  `manifest.csv` is the number the GUI prints for that exact file. A file is
  included only if the verdict is correct at the fixed 0.50 threshold **and**
  `|P - 0.50| >= 0.15`, which is the band where the GUI stops adding its
  "close to the threshold — treat as inconclusive" caveat. All 130 were
  re-verified in a final pass.

Images are full video frames at JPEG quality 95, not pre-cropped faces, so the
face detector runs on them as it would on anything a user uploads.

## manifest.csv

| column | meaning |
|---|---|
| `file` | path within this folder |
| `kind` | `image` or `video` |
| `truth` | `fake` (Celeb-synthesis) or `real` (Celeb-real) |
| `p_manipulated` | the model's P(manipulated) for this file |
| `confidence` | `p` for fakes, `1-p` for reals |
| `verdict` | what the GUI prints |
| `frames_scored` | frames with a detected face / frames sampled |
| `source_video`, `source_folder`, `source_frame` | Celeb-DF-v2 provenance |

## Caveat

This is a selected set: it shows the detector working, not how often it works.
The honest rates are in `results/RESULTS.md` — 0.9276 frame balanced accuracy
and 0.9579 video balanced accuracy for this front end on this split. Building
this folder, 33 clips were scored at video level and 3 came out wrong (2 real
clips called manipulated, 1 fake called authentic); image frames were rejected
at a similar rate. Those rejects are the other side of the same model.
