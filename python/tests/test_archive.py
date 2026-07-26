# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Archives that refuse before they extract, and cleanup that keeps what matters.

**The security tests are first because unsafe extraction is the risk.** A ZIP is
a list of paths, and `extractall` *sanitises* absolute paths and `..` rather than
refusing them — so a hostile entry becomes a real file somewhere unexpected
instead of an error. An archive here can arrive from somebody else, which is the
same trust boundary `.mdp` has, so every one of those cases is a refusal with a
test behind it.

The second half is the ordering: **verify, then delete.** The failure nobody
recovers from is a run archived, removed, and then found unreadable.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from md import archive, library
from test_library import make_run


def a_run(tmp_path: Path, **kwargs: object) -> library.Run:
    path = make_run(tmp_path, "runs-7", **kwargs)  # type: ignore[arg-type]
    run = library.load_run(path)
    assert run is not None
    return run


def rewrite(source: Path, destination: Path, **entries: bytes) -> Path:
    """A copy of ``source`` with entries added or replaced.

    Hand-built rather than produced by `create_archive`, because every case here
    is a file `create_archive` would never write — which is exactly the set a
    reader has to survive.
    """
    with zipfile.ZipFile(source) as original:
        keep = {info.filename: original.read(info.filename) for info in original.infolist()}
    keep.update(entries)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in keep.items():
            out.writestr(name, data)
    return destination


def with_manifest(source: Path, destination: Path, mutate) -> Path:  # noqa: ANN001
    with zipfile.ZipFile(source) as original:
        manifest = json.loads(original.read(archive.MANIFEST_NAME).decode("utf-8"))
    mutate(manifest)
    return rewrite(
        source, destination, **{archive.MANIFEST_NAME: json.dumps(manifest).encode("utf-8")}
    )


# ---- the refusals ------------------------------------------------------------


def test_an_entry_that_escapes_the_archive_is_refused(tmp_path: Path) -> None:
    """The traversal case. `extractall` would rewrite this and write the file.

    Silently landing `etc/passwd` in the restore directory instead of `/etc` is
    better than the alternative and is still not an outcome anyone asked for.
    """
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = with_manifest(
        good,
        tmp_path / "escape.zip",
        lambda m: m["entries"].append({"name": "../../etc/passwd", "size": 0, "sha256": ""}),
    )
    with pytest.raises(archive.ArchiveError, match=r"\.\."):
        archive.verify_archive(broken)


def test_an_absolute_entry_is_refused(tmp_path: Path) -> None:
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = with_manifest(
        good,
        tmp_path / "absolute.zip",
        lambda m: m["entries"].append({"name": "/etc/shadow", "size": 0, "sha256": ""}),
    )
    with pytest.raises(archive.ArchiveError, match="absolute"):
        archive.verify_archive(broken)


def test_a_windows_drive_letter_is_refused(tmp_path: Path) -> None:
    """`C:evil` is absolute on Windows and looks relative to a POSIX check."""
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = with_manifest(
        good,
        tmp_path / "drive.zip",
        lambda m: m["entries"].append({"name": "C:evil.txt", "size": 0, "sha256": ""}),
    )
    with pytest.raises(archive.ArchiveError, match="drive"):
        archive.verify_archive(broken)


def test_a_duplicate_entry_is_refused(tmp_path: Path) -> None:
    """Which of two identically named entries wins is a property of the
    extractor, not of the archive — so an archive may not contain both."""
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    with zipfile.ZipFile(good) as original:
        first = next(i.filename for i in original.infolist() if i.filename != archive.MANIFEST_NAME)
    broken = with_manifest(
        good,
        tmp_path / "dupe.zip",
        lambda m: m["entries"].append({"name": first, "size": 0, "sha256": ""}),
    )
    with pytest.raises(archive.ArchiveError, match="twice"):
        archive.verify_archive(broken)


def test_an_implausible_declared_size_is_refused_before_it_is_decompressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zip bomb is a small file claiming to be a huge one.

    The claim is in the header, where it can be checked for free — so the check
    happens there, before anything is inflated.
    """
    monkeypatch.setattr(archive, "MAX_ENTRY_BYTES", 8)
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    with pytest.raises(archive.ArchiveError, match="implausible"):
        archive.verify_archive(good)


def test_a_corrupt_entry_is_caught_by_its_checksum(tmp_path: Path) -> None:
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = rewrite(good, tmp_path / "corrupt.zip", **{"metrics.csv": b"not what was archived"})
    with pytest.raises(archive.ArchiveError, match="checksum"):
        archive.verify_archive(broken)


def test_a_symlink_entry_is_refused(tmp_path: Path) -> None:
    """`extractall` writes the link *target* as a file — a quiet surprise.

    The type bits are only meaningful when set: `zipfile.writestr` stores
    permissions with no type at all, and a first version of this check rejected
    every such entry, which hid the two checks behind it.
    """
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    linked = tmp_path / "link.zip"
    with zipfile.ZipFile(good) as original, zipfile.ZipFile(linked, "w") as out:
        for info in original.infolist():
            out.writestr(info, original.read(info.filename))
        info = zipfile.ZipInfo("evil-link")
        info.external_attr = 0o120777 << 16  # S_IFLNK
        out.writestr(info, "/etc/passwd")
    with pytest.raises(archive.ArchiveError, match="not a regular file"):
        archive.verify_archive(linked)


def test_an_entry_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    """Otherwise an archive could smuggle a file past the checksums entirely."""
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = rewrite(good, tmp_path / "extra.zip", **{"surprise.sh": b"rm -rf /"})
    with pytest.raises(archive.ArchiveError, match="not in the manifest"):
        archive.verify_archive(broken)


def test_a_manifest_entry_with_no_file_is_refused(tmp_path: Path) -> None:
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")
    broken = with_manifest(
        good,
        tmp_path / "absent.zip",
        lambda m: m["entries"].append({"name": "ghost.csv", "size": 1, "sha256": "0" * 64}),
    )
    with pytest.raises(archive.ArchiveError, match="absent"):
        archive.verify_archive(broken)


def test_an_archive_from_a_future_version_is_refused_by_name(tmp_path: Path) -> None:
    good = archive.create_archive(a_run(tmp_path), tmp_path / "good.zip")

    def bump(manifest: dict[str, object]) -> None:
        manifest["version"] = archive.ARCHIVE_VERSION + 1

    broken = with_manifest(good, tmp_path / "future.zip", bump)
    with pytest.raises(archive.ArchiveError, match="version"):
        archive.verify_archive(broken)


def test_something_that_is_not_an_archive_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("hello", encoding="utf-8")
    with pytest.raises(archive.ArchiveError):
        archive.verify_archive(junk)

    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as out:
        out.writestr("a.txt", "a")
    with pytest.raises(archive.ArchiveError, match="not one of ours"):
        archive.verify_archive(plain)


# ---- the round trip ----------------------------------------------------------


def test_an_archived_run_comes_back(tmp_path: Path) -> None:
    run = a_run(tmp_path, evals={100: 4321.0}, checkpoints=(100, 200), recordings=(100,))
    written = archive.create_archive(run, tmp_path / "runs-7.zip")
    manifest = archive.verify_archive(written)
    assert manifest.run_id == "runs-7"

    restored = archive.restore_archive(written, tmp_path / "restored")
    back = library.load_run(restored)
    assert back is not None
    # The same curves the console drew from the original.
    assert back.best_score == 4321.0
    assert back.checkpoints == 2
    assert back.recordings == 1


def test_restoring_over_something_that_exists_is_refused(tmp_path: Path) -> None:
    """A half-merged run has a metrics.csv from one run and checkpoints from
    another, and nothing downstream would notice."""
    run = a_run(tmp_path)
    written = archive.create_archive(run, tmp_path / "runs-7.zip")
    (tmp_path / "occupied").mkdir()
    with pytest.raises(archive.ArchiveError, match="already exists"):
        archive.restore_archive(written, tmp_path / "occupied")


def test_a_refused_restore_leaves_nothing_behind(tmp_path: Path) -> None:
    run = a_run(tmp_path)
    good = archive.create_archive(run, tmp_path / "good.zip")
    broken = rewrite(good, tmp_path / "corrupt.zip", **{"metrics.csv": b"wrong"})
    with pytest.raises(archive.ArchiveError):
        archive.restore_archive(broken, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("*.incoming"))


def test_a_display_name_and_note_survive_the_round_trip(tmp_path: Path) -> None:
    path = make_run(tmp_path, "runs-9")
    library.rename(path, "amber-anvil")
    library.set_note(path, "the one that learned to wait")
    run = library.load_run(path)
    assert run is not None

    written = archive.create_archive(run, tmp_path / "a.zip")
    assert archive.verify_archive(written).display_name == "amber-anvil"
    back = library.load_run(archive.restore_archive(written, tmp_path / "back"))
    assert back is not None
    assert back.display_name == "amber-anvil"
    assert back.note == "the one that learned to wait"


# ---- nothing is deleted before verification ----------------------------------


def test_the_original_is_only_removed_after_the_archive_verifies(tmp_path: Path) -> None:
    """The ordering the whole module exists to guarantee."""
    run = a_run(tmp_path, checkpoints=(100,), recordings=(100,))
    written, freed = archive.archive_and_remove(run, tmp_path / "out.zip", root=tmp_path)
    assert written.is_file()
    assert not run.path.exists()
    assert freed > 0
    # And it really did come back.
    assert library.load_run(archive.restore_archive(written, tmp_path / "back")) is not None


def test_a_run_outside_the_managed_root_is_never_removed(tmp_path: Path) -> None:
    """A run directory arrives from a picker, an env var or a command line."""
    run = a_run(tmp_path)
    with pytest.raises(archive.ArchiveError, match="outside"):
        archive.archive_and_remove(run, tmp_path / "out.zip", root=tmp_path / "somewhere-else")
    assert run.path.exists()


# ---- cleanup -----------------------------------------------------------------


def test_cleanup_keeps_the_summary_the_best_and_the_pinned(tmp_path: Path) -> None:
    """Three things you cannot get back, and each is protected by name."""
    path = make_run(
        tmp_path,
        "runs-7",
        evals={100: 1.0, 300: 90.0},
        checkpoints=(100, 200, 300),
        recordings=(100, 200, 300),
    )
    library.pin(path, path / "update-00200.mdr")
    (path / "checkpoints" / "policy-best.pt").write_bytes(b"x" * 10)
    run = library.load_run(path)
    assert run is not None

    plan = archive.plan_cleanup(run)
    kept = {p.name for p in plan.keep}
    removed = {p.name for p in plan.remove}

    assert "metrics.csv" in kept and "evals.csv" in kept
    assert "policy-00300.pt" in kept  # the best *evaluated* checkpoint
    assert "policy-best.pt" in kept
    assert "update-00200.mdr" in kept  # pinned
    assert "policy-00100.pt" in removed
    assert "update-00100.mdr" in removed
    assert plan.reclaim_bytes > 0


def test_a_plan_is_executed_exactly_as_shown(tmp_path: Path) -> None:
    """A button that recomputed at the moment of the click would be offering a
    different answer from the one that was agreed to."""
    path = make_run(tmp_path, "runs-7", evals={100: 1.0}, checkpoints=(100, 200), recordings=(100,))
    run = library.load_run(path)
    assert run is not None
    plan = archive.plan_cleanup(run)
    doomed = list(plan.remove)

    freed = archive.apply_cleanup(plan, tmp_path)
    assert freed == plan.reclaim_bytes
    for gone in doomed:
        assert not gone.exists()
    for kept in plan.keep:
        assert kept.exists()


def test_cleanup_refuses_a_path_outside_the_root(tmp_path: Path) -> None:
    """A plan is data; it can be stale or forged."""
    outside = tmp_path / "elsewhere" / "victim.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    plan = archive.CleanupPlan(keep=(), remove=(outside,), reclaim_bytes=1)
    with pytest.raises(archive.ArchiveError, match="outside"):
        archive.apply_cleanup(plan, tmp_path / "runs")
    assert outside.exists()


def test_a_run_with_nothing_spare_offers_nothing(tmp_path: Path) -> None:
    path = make_run(tmp_path, "runs-7", evals={100: 1.0}, checkpoints=(100,))
    run = library.load_run(path)
    assert run is not None
    plan = archive.plan_cleanup(run)
    assert plan.empty
    assert archive.describe(plan) == "nothing to remove"


def test_the_plan_says_what_it_would_free_before_it_frees_it(tmp_path: Path) -> None:
    path = make_run(tmp_path, "runs-7", evals={100: 1.0}, checkpoints=(100, 200, 300))
    run = library.load_run(path)
    assert run is not None
    line = archive.describe(archive.plan_cleanup(run))
    assert "files" in line
    assert "kB" in line or "MB" in line or "B" in line
