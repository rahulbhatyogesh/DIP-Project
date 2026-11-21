"""Tests for face extraction: cropping geometry, frame sampling, and the
end-to-end video -> crops -> manifest path.

Uses a synthetic MP4 and a stub face detector, so it runs anywhere without
mediapipe or the 20 GB dataset.

Run:  python -m pytest tests/test_extract.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import extract_faces as ef  # noqa: E402


def _fourcc(code: str) -> int:
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return int(fn(*code))


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """A 40-frame 320x240 clip with a bright square standing in for a face."""
    path = tmp_path / "Celeb-real" / "id0_0000.mp4"
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), _fourcc("mp4v"), 10.0, (320, 240))
    assert writer.isOpened(), "could not open VideoWriter"
    for t in range(40):
        frame = np.full((240, 320, 3), 40, np.uint8)
        x = 100 + t  # drifts, so a tracker has something to follow
        frame[80:160, x : x + 80] = 220
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def stub_detector():
    """Install a detector that always returns a fixed centre box."""
    ef._DETECTOR = lambda rgb: (100.0, 80.0, 180.0, 160.0)
    ef._DETECTOR_KIND = "stub"
    yield
    ef._DETECTOR = None
    ef._DETECTOR_KIND = None


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------

def test_sampling_is_evenly_spaced_and_in_range():
    idx = ef.sample_frame_indices(390, 32)
    assert len(idx) == 32
    assert idx[0] == 0 and idx[-1] == 389
    assert np.all(np.diff(idx) > 0)


def test_sampling_takes_everything_from_short_clips():
    assert ef.sample_frame_indices(20, 32).tolist() == list(range(20))


def test_sampling_handles_unreadable_frame_count():
    assert len(ef.sample_frame_indices(0, 32)) == 0


# --------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------

def test_crop_is_square_and_right_size():
    frame = np.zeros((240, 320, 3), np.uint8)
    out = ef.square_crop(frame, (100, 80, 180, 160), margin=1.3, size=224)
    assert out.shape == (224, 224, 3)


def test_crop_is_centred_on_the_face_box():
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[80:160, 100:180] = 255           # the "face"
    out = ef.square_crop(frame, (100, 80, 180, 160), margin=1.0, size=64)
    assert np.all(out[32, 32] == 255)      # centre of crop is inside the face
    assert out.mean() > 200                # margin=1.0 => crop is nearly all face


def test_margin_adds_context_around_the_face():
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[80:160, 100:180] = 255
    tight = ef.square_crop(frame, (100, 80, 180, 160), margin=1.0, size=64)
    loose = ef.square_crop(frame, (100, 80, 180, 160), margin=1.6, size=64)
    assert loose.mean() < tight.mean()     # more dark background pulled in


def test_face_at_border_is_padded_not_squashed():
    """A box running off the edge must stay square via replicate-padding."""
    frame = np.full((240, 320, 3), 128, np.uint8)
    out = ef.square_crop(frame, (0, 0, 60, 60), margin=2.0, size=224)
    assert out.shape == (224, 224, 3)
    assert np.all(out == 128)


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def test_extract_one_writes_crops_and_manifest_rows(synthetic_video, stub_detector, tmp_path):
    out = tmp_path / "faces"
    rec = ("id0_0000", "Celeb-real", 0, "id0", "id0", "Celeb-real/id0_0000.mp4")
    res = ef.extract_one(
        rec, root=str(synthetic_video.parents[1]), out=str(out),
        n_frames=8, size=224, margin=1.3, detector_kind="stub", jpeg_quality=95,
    )

    assert res["status"] == "ok"
    assert len(res["rows"]) == 8

    jpgs = sorted((out / "Celeb-real" / "id0_0000").glob("*.jpg"))
    assert len(jpgs) == 8
    img = cv2.imread(str(jpgs[0]))
    assert img.shape == (224, 224, 3)

    row = res["rows"][0]
    assert row["label"] == 0 and row["source_id"] == "id0"
    assert row["rel_path"].startswith("Celeb-real/id0_0000/")


def test_rerun_is_cached_not_redone(synthetic_video, stub_detector, tmp_path):
    out = tmp_path / "faces"
    rec = ("id0_0000", "Celeb-real", 0, "id0", "id0", "Celeb-real/id0_0000.mp4")
    job = dict(
        root=str(synthetic_video.parents[1]), out=str(out), n_frames=8,
        size=224, margin=1.3, detector_kind="stub", jpeg_quality=95,
    )
    assert ef.extract_one(rec, **job)["status"] == "ok"
    second = ef.extract_one(rec, **job)
    assert second["status"] == "cached"
    assert len(second["rows"]) == 8


def test_undetectable_face_is_reported_not_crashed(synthetic_video, tmp_path):
    ef._DETECTOR = lambda rgb: None
    ef._DETECTOR_KIND = "stub"
    try:
        res = ef.extract_one(
            ("id0_0000", "Celeb-real", 0, "id0", "id0", "Celeb-real/id0_0000.mp4"),
            root=str(synthetic_video.parents[1]), out=str(tmp_path / "faces"),
            n_frames=8, size=224, margin=1.3, detector_kind="stub", jpeg_quality=95,
        )
        assert res["status"] == "no_face"
        assert res["rows"] == []
    finally:
        ef._DETECTOR = None
        ef._DETECTOR_KIND = None


def test_missing_video_is_reported_not_crashed(tmp_path, stub_detector):
    res = ef.extract_one(
        ("ghost", "Celeb-real", 0, "id9", "id9", "Celeb-real/ghost.mp4"),
        root=str(tmp_path), out=str(tmp_path / "faces"),
        n_frames=8, size=224, margin=1.3, detector_kind="stub", jpeg_quality=95,
    )
    assert res["status"] in ("unreadable", "error")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
