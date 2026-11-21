"""Tests for the Celeb-DF-v2 split logic.

Builds a synthetic directory tree with Celeb-DF's exact naming conventions and
Table I's exact counts (590 real / 5639 fake / 300 YouTube-real, 59 celebrity
subjects), so the split rules can be verified without the 20 GB download.

Run:  python -m pytest tests/test_splits.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import (  # noqa: E402
    FAKE,
    REAL,
    build_splits,
    build_splits_from_manifest,
    identity_components,
    make_identity_split,
    make_random_split,
    make_strict_identity_split,
    parse_video,
    records_from_manifest,
    scan_dataset,
    summarise,
)

N_IDENTITIES = 59
N_REAL = 590
N_FAKE = 5639
N_YT = 300


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> Path:
    """A synthetic Celeb-DF-v2 tree matching the paper's Table I counts."""
    root = tmp_path_factory.mktemp("celeb-df-v2")
    rng = random.Random(0)

    real_dir = root / "Celeb-real"
    fake_dir = root / "Celeb-synthesis"
    yt_dir = root / "YouTube-real"
    for d in (real_dir, fake_dir, yt_dir):
        d.mkdir()

    # 590 reals spread over 59 subjects (10 each).
    for i in range(N_IDENTITIES):
        for k in range(N_REAL // N_IDENTITIES):
            (real_dir / f"id{i}_{k:04d}.mp4").touch()

    # 5639 fakes: source face I rendered onto target video J, I != J.
    made = 0
    k = 0
    while made < N_FAKE:
        i = rng.randrange(N_IDENTITIES)
        j = rng.randrange(N_IDENTITIES)
        if i == j:
            continue
        p = fake_dir / f"id{i}_id{j}_{k:04d}.mp4"
        if p.exists():
            k += 1
            continue
        p.touch()
        made += 1
        k += 1

    # 300 YouTube-real, one ordinary subject each.
    for n in range(N_YT):
        (yt_dir / f"{n:05d}.mp4").touch()

    return root


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

def test_parse_real():
    r = parse_video("Celeb-real/id17_0008.mp4")
    assert (r.label, r.source_id, r.target_id) == (REAL, "id17", "id17")


def test_parse_fake_source_is_first_id():
    """id28_id17_0008 = subject 28's face rendered onto subject 17's video."""
    r = parse_video("Celeb-synthesis/id28_id17_0008.mp4")
    assert r.label == FAKE
    assert r.source_id == "id28"   # face actually rendered
    assert r.target_id == "id17"   # video/background donor
    assert r.identity_key("source") == "id28"
    assert r.identity_key("target") == "id17"


def test_parse_youtube_real_is_its_own_identity():
    a = parse_video("YouTube-real/00042.mp4")
    b = parse_video("YouTube-real/00043.mp4")
    assert a.label == REAL
    assert a.identity_key() != b.identity_key()


def test_rejects_unknown_folder():
    with pytest.raises(ValueError):
        parse_video("Nonsense/id1_0000.mp4")


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def test_scan_excludes_youtube_real_by_default(fake_root):
    recs = scan_dataset(fake_root)
    assert sum(1 for r in recs if r.label == REAL) == N_REAL   # Table I: 590
    assert sum(1 for r in recs if r.label == FAKE) == N_FAKE   # Table I: 5639
    assert len({r.identity_key() for r in recs}) == N_IDENTITIES


def test_scan_can_include_youtube_real(fake_root):
    recs = scan_dataset(fake_root, include_youtube_real=True)
    assert sum(1 for r in recs if r.label == REAL) == N_REAL + N_YT


# --------------------------------------------------------------------------
# The identity split -- the rule under test
# --------------------------------------------------------------------------

def test_identity_split_has_no_key_identity_overlap(fake_root):
    recs = scan_dataset(fake_root)
    ident = {r.video_id: r.identity_key("source") for r in recs}
    splits = make_identity_split(recs)
    seen = {part: {ident[v] for v in vids} for part, vids in splits.items()}
    assert not (seen["train"] & seen["test"])
    assert not (seen["train"] & seen["val"])
    assert not (seen["val"] & seen["test"])


def test_person_a_reals_and_fakes_travel_together(fake_root):
    """A subject's own real videos and the fakes keyed to them land in one partition."""
    recs = scan_dataset(fake_root)
    splits = make_identity_split(recs)
    part_of = {v: part for part, vids in splits.items() for v in vids}
    by_identity: dict[str, set[str]] = {}
    for r in recs:
        by_identity.setdefault(r.identity_key("source"), set()).add(part_of[r.video_id])
    for ident, parts in by_identity.items():
        assert len(parts) == 1, f"{ident} was scattered across {parts}"


def test_secondary_identity_may_cross_partitions(fake_root):
    """By design: only the KEY identity is constrained, not the other one."""
    recs = scan_dataset(fake_root)
    splits = make_identity_split(recs, key="source")
    part_of = {v: part for part, vids in splits.items() for v in vids}
    crossed = any(
        part_of[r.video_id] != part_of.get(f"{r.target_id}_0000", part_of[r.video_id])
        for r in recs
        if r.label == FAKE
    )
    assert crossed, "expected target identities to span partitions under source keying"


def test_identity_split_drops_nothing(fake_root):
    recs = scan_dataset(fake_root)
    splits = make_identity_split(recs)
    assigned = [v for vids in splits.values() for v in vids]
    assert len(assigned) == len(recs)
    assert set(assigned) == {r.video_id for r in recs}


def test_identity_split_honours_requested_fractions(fake_root):
    recs = scan_dataset(fake_root)
    ident = {r.video_id: r.identity_key() for r in recs}
    splits = make_identity_split(recs)
    n_test_ids = len({ident[v] for v in splits["test"]})
    n_val_ids = len({ident[v] for v in splits["val"]})
    # Paper: 21% of identities to test, 19% to val.
    assert abs(n_test_ids / N_IDENTITIES - 0.21) < 0.03
    assert abs(n_val_ids / N_IDENTITIES - 0.19) < 0.03


def test_target_keying_is_a_distinct_split(fake_root):
    recs = scan_dataset(fake_root)
    assert make_identity_split(recs, key="source") != make_identity_split(recs, key="target")


def test_every_identity_partition_contains_both_classes(fake_root):
    """Otherwise a partition would be unusable for binary training/eval."""
    recs = scan_dataset(fake_root)
    label = {r.video_id: r.label for r in recs}
    for part, vids in make_identity_split(recs).items():
        labels = {label[v] for v in vids}
        assert labels == {REAL, FAKE}, f"{part} only has labels {labels}"


# --------------------------------------------------------------------------
# The random split -- must leak, that is the point of Table II
# --------------------------------------------------------------------------

def test_random_split_does_leak_identities(fake_root):
    """The contrast that produces the paper's Table II accuracy inflation."""
    recs = scan_dataset(fake_root)
    ident = {r.video_id: r.identity_key() for r in recs}
    splits = make_random_split(recs)
    train_ids = {ident[v] for v in splits["train"]}
    test_ids = {ident[v] for v in splits["test"]}
    overlap = train_ids & test_ids
    assert len(overlap) > 0.9 * N_IDENTITIES, "random split should leak nearly every identity"


def test_random_split_drops_nothing_and_is_stratified(fake_root):
    recs = scan_dataset(fake_root)
    label = {r.video_id: r.label for r in recs}
    splits = make_random_split(recs)
    assigned = [v for vids in splits.values() for v in vids]
    assert len(assigned) == len(recs)
    fake_frac = sum(label[v] for v in splits["test"]) / len(splits["test"])
    assert abs(fake_frac - N_FAKE / (N_FAKE + N_REAL)) < 0.02


def test_splits_are_deterministic(fake_root):
    recs = scan_dataset(fake_root)
    assert make_identity_split(recs, seed=7) == make_identity_split(recs, seed=7)
    assert make_identity_split(recs, seed=7) != make_identity_split(recs, seed=8)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def test_build_splits_writes_usable_manifest(fake_root, tmp_path):
    out = tmp_path / "splits.json"
    m = build_splits(fake_root, out)
    assert out.exists()
    assert (m["n_real"], m["n_fake"], m["n_identities"]) == (N_REAL, N_FAKE, N_IDENTITIES)
    assert {"random", "identity"} <= set(m["splits"])
    assert "identity overlap" in summarise(m)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# --------------------------------------------------------------------------
# Strict component split (our extension, not the paper's protocol)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clustered_root(tmp_path_factory) -> Path:
    """A tree whose identity graph has three disjoint components.

    Celeb-DF-v2 really does split into components of size [38, 10, 10, 1], but a
    uniformly random pairing collapses into one, so the main fixture cannot
    exercise the strict split.
    """
    root = tmp_path_factory.mktemp("clustered")
    (root / "Celeb-real").mkdir()
    (root / "Celeb-synthesis").mkdir()

    clusters = [list(range(0, 20)), list(range(20, 32)), list(range(32, 40))]
    for cluster in clusters:
        for i in cluster:
            for k in range(5):
                (root / "Celeb-real" / f"id{i}_{k:04d}.mp4").touch()
        # forgeries only ever pair identities inside the same cluster
        for a in cluster:
            for b in cluster:
                if a != b:
                    (root / "Celeb-synthesis" / f"id{a}_id{b}_0000.mp4").touch()
    return root


def test_identity_graph_finds_the_components(clustered_root):
    comps = identity_components(scan_dataset(clustered_root))
    assert sorted(len(c) for c in comps) == [8, 12, 20]


def test_strict_split_isolates_every_identity_in_both_roles(clustered_root):
    """The whole point: no identity appears in two partitions as source OR target."""
    recs = scan_dataset(clustered_root)
    splits = make_strict_identity_split(recs)
    part_of = {v: p for p, vids in splits.items() for v in vids}

    seen: dict[str, set[str]] = {}
    for r in recs:
        for ident in (r.source_id, r.target_id):
            if ident:
                seen.setdefault(ident, set()).add(part_of[r.video_id])
    scattered = {i: p for i, p in seen.items() if len(p) > 1}
    assert not scattered, f"identities spanning partitions: {scattered}"


def test_strict_split_is_stricter_than_the_paper_protocol(clustered_root):
    """Contrast: the paper's per-identity split DOES leak the secondary identity."""
    recs = scan_dataset(clustered_root)
    loose = make_identity_split(recs, key="source")
    part_of = {v: p for p, vids in loose.items() for v in vids}

    seen: dict[str, set[str]] = {}
    for r in recs:
        for ident in (r.source_id, r.target_id):
            if ident:
                seen.setdefault(ident, set()).add(part_of[r.video_id])
    assert any(len(p) > 1 for p in seen.values()), \
        "expected target-identity leakage under source keying"


def test_strict_split_drops_nothing(clustered_root):
    recs = scan_dataset(clustered_root)
    assigned = [v for vids in make_strict_identity_split(recs).values() for v in vids]
    assert sorted(assigned) == sorted(r.video_id for r in recs)


# --------------------------------------------------------------------------
# Rebuilding splits from a manifest (no source video needed)
# --------------------------------------------------------------------------

def test_records_from_manifest_round_trips(fake_root, tmp_path):
    import csv as _csv

    recs = scan_dataset(fake_root)
    man = tmp_path / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=[
            "video_id", "folder", "label", "source_id", "target_id",
            "frame_idx", "rel_path",
        ])
        w.writeheader()
        for r in recs:
            for f in range(2):          # two rows per video, to test dedup
                w.writerow({
                    "video_id": r.video_id, "folder": r.folder, "label": r.label,
                    "source_id": r.source_id or "", "target_id": r.target_id or "",
                    "frame_idx": f, "rel_path": f"{r.folder}/{r.video_id}/{f}.jpg",
                })

    rebuilt = records_from_manifest(man)
    assert len(rebuilt) == len(recs)
    assert {r.video_id for r in rebuilt} == {r.video_id for r in recs}
    by_id = {r.video_id: r for r in rebuilt}
    for r in recs:
        assert by_id[r.video_id].label == r.label
        assert by_id[r.video_id].source_id == r.source_id

    m = build_splits_from_manifest(man, tmp_path / "s.json")
    assert {"random", "identity"} <= set(m["splits"])
    assert m["n_real"] == N_REAL and m["n_fake"] == N_FAKE


def test_build_splits_skips_strict_when_infeasible(fake_root, tmp_path):
    """Uniform random pairing gives one giant component -- no strict split exists."""
    m = build_splits(fake_root, tmp_path / "s.json")
    assert set(m["splits"]) == {"random", "identity"}
    assert "identity_strict_unavailable" in m
    assert "component" in m["identity_strict_unavailable"]


def test_build_splits_emits_strict_when_feasible(clustered_root, tmp_path):
    m = build_splits(clustered_root, tmp_path / "s.json")
    assert set(m["splits"]) == {"random", "identity", "identity_strict"}
    assert "identity_strict_unavailable" not in m


def test_strict_split_raises_rather_than_emptying_a_partition(fake_root):
    with pytest.raises(ValueError, match="component"):
        make_strict_identity_split(scan_dataset(fake_root))
