"""Train/val/test splits for Celeb-DF-v2 -- random (by video) vs identity (by subject).

This is what produces the paper's Table II result: the same model scores ~6-8
accuracy points higher under a random split purely because subject identities
leak across the partitions.

Filename conventions
--------------------
    Celeb-real/id{J}_{K}.mp4             real, subject J
    Celeb-synthesis/id{I}_id{J}_{K}.mp4  fake, subject I's face rendered onto
                                         subject J's video K
    YouTube-real/{N}.mp4                 real, one unique ordinary subject each

That id{J}_{K}.mp4 is itself a valid Celeb-real filename is what tells us J
donates the video/background and I donates the face.

Identity-split rule
-------------------
A fake involves two subjects, but they need NOT land in the same partition.
Each video is keyed to exactly ONE identity and follows that identity:

    person A's real videos and A's fakes always travel together,
    and person B is assigned independently of whether B ever appeared with A.

Nothing is dropped and every subject's own footage stays on one side of the
split.  The key defaults to the SOURCE identity -- the face actually rendered
into the frame, hence the one carrying the artifacts being classified.
Passing key="target" exposes the alternative for a sensitivity check.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REAL, FAKE = 0, 1

FOLDER_REAL = "Celeb-real"
FOLDER_FAKE = "Celeb-synthesis"
FOLDER_YT = "YouTube-real"

_RE_REAL = re.compile(r"^id(\d+)_(\d+)$")
_RE_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d+)$")

# Paper, Sec. V-A: "using 21% of the identities for testing and 19% for validation".
DEFAULT_FRACS = {"train": 0.60, "val": 0.19, "test": 0.21}


@dataclass(frozen=True)
class VideoRecord:
    rel_path: str
    video_id: str
    folder: str
    label: int
    source_id: str | None   # whose face is rendered
    target_id: str | None   # whose video/background is used

    def identity_key(self, key: str = "source") -> str:
        if key not in ("source", "target"):
            raise ValueError("key must be 'source' or 'target'")
        chosen = self.source_id if key == "source" else self.target_id
        # Reals carry a single identity; YouTube-real videos are their own subject.
        return chosen or self.source_id or self.target_id or f"yt_{self.video_id}"


def parse_video(rel_path: str) -> VideoRecord:
    """Parse one Celeb-DF path like 'Celeb-synthesis/id0_id16_0000.mp4'."""
    p = Path(rel_path.replace("\\", "/"))
    folder, stem = p.parent.name, p.stem

    if folder == FOLDER_FAKE:
        m = _RE_FAKE.match(stem)
        if not m:
            raise ValueError(f"unparseable fake filename: {rel_path}")
        src, tgt, _ = m.groups()
        return VideoRecord(rel_path, stem, folder, FAKE, f"id{src}", f"id{tgt}")

    if folder == FOLDER_REAL:
        m = _RE_REAL.match(stem)
        if not m:
            raise ValueError(f"unparseable real filename: {rel_path}")
        subj, _ = m.groups()
        return VideoRecord(rel_path, stem, folder, REAL, f"id{subj}", f"id{subj}")

    if folder == FOLDER_YT:
        # No id prefix, but 300 videos / 300 subjects => each file is its own identity.
        return VideoRecord(rel_path, stem, folder, REAL, None, None)

    raise ValueError(f"unexpected folder {folder!r} in {rel_path}")


def scan_dataset(root: str | Path, include_youtube_real: bool = False) -> list[VideoRecord]:
    """Walk a Celeb-DF-v2 root and parse every .mp4.

    YouTube-real is EXCLUDED by default.  No fakes are ever generated from those
    300 subjects and the footage comes from a visibly different source
    distribution, so including it lets a model learn "this source bucket => real"
    without learning any forgery artifact -- the same shortcut the paper attacks
    in Sec. V-A, keyed on source domain instead of identity.  Excluding it also
    reproduces Table I's count of 590 real videos.
    """
    root = Path(root)
    folders = [FOLDER_REAL, FOLDER_FAKE] + ([FOLDER_YT] if include_youtube_real else [])
    records: list[VideoRecord] = []
    for folder in folders:
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"missing expected folder: {d}")
        for mp4 in sorted(d.glob("*.mp4")):
            records.append(parse_video(f"{folder}/{mp4.name}"))
    if not records:
        raise RuntimeError(f"no videos found under {root}")
    return records


def _partition(items: list, fracs: dict[str, float], rng: random.Random) -> dict[str, list]:
    """Shuffle and cut into train/val/test by fraction, losing nothing."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = round(n * fracs["val"])
    n_test = round(n * fracs["test"])
    return {
        "test": items[:n_test],
        "val": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }


def make_random_split(
    records: list[VideoRecord],
    fracs: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split by VIDEO, stratified by label.  The paper's (leaky) baseline protocol."""
    fracs = fracs or DEFAULT_FRACS
    rng = random.Random(seed)
    out: dict[str, list[str]] = defaultdict(list)
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r.video_id)
    for label in sorted(by_label):
        for part, vids in _partition(by_label[label], fracs, rng).items():
            out[part].extend(vids)
    return {k: sorted(v) for k, v in out.items()}


def make_identity_split(
    records: list[VideoRecord],
    fracs: dict[str, float] | None = None,
    seed: int = 42,
    key: str = "source",
) -> dict[str, list[str]]:
    """Split by SUBJECT.  Every video follows its key identity; nothing is dropped."""
    fracs = fracs or DEFAULT_FRACS
    rng = random.Random(seed)

    by_identity: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_identity[r.identity_key(key)].append(r.video_id)

    identities = sorted(by_identity)
    parts = _partition(identities, fracs, rng)
    return {
        part: sorted(v for ident in idents for v in by_identity[ident])
        for part, idents in parts.items()
    }


def build_splits(
    root: str | Path,
    out_path: str | Path,
    *,
    seed: int = 42,
    identity_key: str = "source",
    include_youtube_real: bool = False,
) -> dict:
    """Produce both protocols in one manifest JSON."""
    records = scan_dataset(root, include_youtube_real)
    label_of = {r.video_id: r.label for r in records}
    ident_of = {r.video_id: r.identity_key(identity_key) for r in records}

    manifest = {
        "seed": seed,
        "identity_key": identity_key,
        "include_youtube_real": include_youtube_real,
        "n_videos": len(records),
        "n_real": sum(1 for r in records if r.label == REAL),
        "n_fake": sum(1 for r in records if r.label == FAKE),
        "n_identities": len(set(ident_of.values())),
        "labels": label_of,
        "identities": ident_of,
        "splits": {
            "random": make_random_split(records, seed=seed),
            "identity": make_identity_split(records, seed=seed, key=identity_key),
        },
        "videos": {r.video_id: asdict(r) for r in records},
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def make_strict_identity_split(
    records: list[VideoRecord],
    fracs: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Strictest reading of the paper: no identity appears in two partitions at all.

    A fake links two subjects, so the constraint propagates: whole connected
    components of the identity graph must move together.  On Celeb-DF-v2 the
    graph has four components of sizes [38, 10, 10, 1], so this is feasible.

    NOT the paper's protocol, though -- it is a stricter extension.  With 59
    identities the paper's "21% for testing and 19% for validation" gives
    round(59*0.21)=12 and round(59*0.19)=11, leaving 36 to train: exactly what
    per-identity assignment (`make_identity_split`) yields.  Whole components
    can only give 38/10/11, i.e. 18.6% and 16.9%, which the authors would not
    have described as 21% and 19%.  So `make_identity_split` is the faithful
    reproduction; this function is for asking what happens with zero leakage.

    Unlike `make_identity_split`, a fake here is excluded from a partition if
    EITHER of its identities belongs elsewhere, so there is no leakage through
    the target subject's background, pose or hair.
    """
    fracs = fracs or DEFAULT_FRACS

    comps = identity_components(records)
    n = sum(len(c) for c in comps)
    targets = {p: fracs[p] * n for p in ("train", "val", "test")}
    assigned: dict[str, set[str]] = {p: set() for p in targets}

    # Largest first, each to whichever partition is furthest below its target.
    # Deterministic: `seed` is accepted only for symmetry with the other splits.
    for comp in sorted(comps, key=len, reverse=True):
        part = max(targets, key=lambda p: targets[p] - len(assigned[p]))
        assigned[part] |= comp

    part_of_ident = {i: p for p, idents in assigned.items() for i in idents}
    out: dict[str, list[str]] = defaultdict(list)
    for r in records:
        idents = [i for i in (r.source_id, r.target_id) if i]
        parts = {part_of_ident[i] for i in idents if i in part_of_ident}
        if len(parts) == 1:
            out[parts.pop()].append(r.video_id)
        elif not parts:
            out["train"].append(r.video_id)        # YouTube-real, no identity
        # len(parts) > 1 cannot happen: linked identities share a component.
    return {k: sorted(v) for k, v in out.items()}


def records_from_manifest(manifest_csv: str | Path) -> list[VideoRecord]:
    """Rebuild VideoRecords from an extraction manifest.

    Lets splits be regenerated anywhere the face crops are (Colab, say) without
    needing the 10 GB of source video.
    """
    import csv as _csv

    seen: dict[str, VideoRecord] = {}
    with Path(manifest_csv).open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            vid = row["video_id"]
            if vid in seen:
                continue
            seen[vid] = VideoRecord(
                rel_path=f"{row['folder']}/{vid}.mp4",
                video_id=vid,
                folder=row["folder"],
                label=int(row["label"]),
                source_id=row["source_id"] or None,
                target_id=row["target_id"] or None,
            )
    if not seen:
        raise RuntimeError(f"no rows in {manifest_csv}")
    return list(seen.values())


def identity_components(records: list[VideoRecord]) -> list[set[str]]:
    """Connected components of the identity graph.

    The paper says only that it puts "real and fake videos of the same person in
    the same set", without saying what to do with a fake that involves TWO
    people.  The strictest reading -- both identities must share a partition --
    is only feasible if the graph (identities as nodes, fakes as edges) breaks
    into several components.  Celeb-DF-v2 synthesises swaps across many pairs of
    its 59 celebrities, so in practice this collapses into one giant component,
    which would make a strict split impossible and forces a single-identity key.

    Run this to confirm that for the actual data rather than assuming it.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in records:
        if r.source_id:
            find(r.source_id)
        if r.target_id:
            find(r.target_id)
        if r.label == FAKE and r.source_id and r.target_id:
            union(r.source_id, r.target_id)

    groups: dict[str, set[str]] = defaultdict(set)
    for ident in parent:
        groups[find(ident)].add(ident)
    return sorted(groups.values(), key=len, reverse=True)


def describe_identity_graph(root: str | Path, include_youtube_real: bool = False) -> str:
    """Report whether a strict both-identities-together split is possible."""
    records = scan_dataset(root, include_youtube_real)
    comps = identity_components(records)
    n_ident = len({i for r in records for i in (r.source_id, r.target_id) if i})
    biggest = len(comps[0]) if comps else 0

    lines = [
        f"identities={n_ident}  components={len(comps)}  largest={biggest}",
        f"component sizes: {[len(c) for c in comps[:10]]}",
    ]
    if biggest == n_ident:
        lines.append(
            "=> ONE component: every identity is linked to every other through some\n"
            "   fake, so requiring both identities of a fake to share a partition puts\n"
            "   the whole dataset in one set. A strict split is impossible, and the\n"
            "   paper must have keyed each video on a single identity."
        )
    else:
        lines.append(
            "=> Multiple components: a strict split IS possible by assigning whole\n"
            "   components to partitions."
        )
    return "\n".join(lines)


def summarise(manifest: dict) -> str:
    """Human-readable split summary, including the identity-leakage contrast."""
    lines = [
        f"videos={manifest['n_videos']}  real={manifest['n_real']}  fake={manifest['n_fake']}"
        f"  identities={manifest['n_identities']}  key={manifest['identity_key']}",
    ]
    labels, idents = manifest["labels"], manifest["identities"]
    for proto, splits in manifest["splits"].items():
        lines.append(f"  [{proto}]")
        seen: dict[str, set] = {}
        for part in ("train", "val", "test"):
            vids = splits[part]
            c = Counter(labels[v] for v in vids)
            seen[part] = {idents[v] for v in vids}
            pct = 100 * len(vids) / manifest["n_videos"]
            lines.append(
                f"    {part:5s} videos={len(vids):5d} ({pct:4.1f}%)  "
                f"real={c[REAL]:4d} fake={c[FAKE]:5d}  identities={len(seen[part]):3d}"
            )
        overlap = seen["train"] & seen["test"]
        lines.append(f"    train/test identity overlap: {len(overlap)}")
    return "\n".join(lines)
