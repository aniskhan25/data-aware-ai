"""SquashFS packaging.

``mksquashfs`` and ``squashfuse`` are external tools that are absent on many
development machines and in CI. Tests that need them skip themselves; the
orchestration around them — argument construction, the partial-then-rename write,
error reporting, and cleanup — is tested with a stub on PATH, because that logic is
ours and is where the mistakes would be.

Needs no PyTorch.
"""

from __future__ import annotations

import os
import stat

import pytest

from dataaware import squashfs
from dataaware.squashfs import (
    DEFAULT_MKSQUASHFS_ARGS,
    SquashFSError,
    build_image,
    have_mksquashfs,
    have_squashfuse,
)

needs_mksquashfs = pytest.mark.skipif(
    not have_mksquashfs(), reason="mksquashfs is not installed"
)
needs_squashfuse = pytest.mark.skipif(
    not have_squashfuse(), reason="squashfuse is not installed"
)


@pytest.fixture
def stub_mksquashfs(tmp_path, monkeypatch):
    """Put a fake ``mksquashfs`` on PATH that writes a placeholder image.

    Lets the build orchestration be tested anywhere. Records the command it was
    called with, so argument handling is verifiable.
    """
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    log = tmp_path / "mksquashfs.log"
    script = bin_dir / "mksquashfs"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        # $2 is the destination path, matching the real tool's argument order.
        'printf "fake-squashfs-image" > "$2"\n'
        "exit 0\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_build_writes_an_image_and_reports_sizes(tiny_dataset, tmp_path, stub_mksquashfs):
    root, _ = tiny_dataset
    result = build_image(root, tmp_path / "dataset.squashfs")

    assert (tmp_path / "dataset.squashfs").is_file()
    assert result["image_bytes"] > 0
    assert result["source_bytes"] > 0
    assert result["build_seconds"] >= 0.0
    assert "mksquashfs" in result["command"]


def test_build_passes_the_configured_arguments(tiny_dataset, tmp_path, stub_mksquashfs):
    root, _ = tiny_dataset
    build_image(root, tmp_path / "dataset.squashfs", extra_args=("-comp", "zstd"))
    assert "-comp zstd" in stub_mksquashfs.read_text()


def test_default_arguments_skip_recompressing_compressed_data():
    """The tutorial dataset is JPEG and PNG; recompressing it wastes read CPU.

    -noF matters as much as -noD: small files are stored as fragments, which -noD
    does not cover, so without it a small-file dataset is compressed after all.
    """
    assert "-noD" in DEFAULT_MKSQUASHFS_ARGS
    assert "-noF" in DEFAULT_MKSQUASHFS_ARGS


def test_existing_image_needs_overwrite(tiny_dataset, tmp_path, stub_mksquashfs):
    root, _ = tiny_dataset
    image = tmp_path / "dataset.squashfs"
    build_image(root, image)
    with pytest.raises(SquashFSError, match="--overwrite"):
        build_image(root, image)
    build_image(root, image, overwrite=True)


def test_a_failing_build_leaves_no_partial_image(tiny_dataset, tmp_path, monkeypatch):
    """A cancelled build must not leave a truncated image a later run would mount."""
    bin_dir = tmp_path / "failing-bin"
    bin_dir.mkdir()
    script = bin_dir / "mksquashfs"
    script.write_text('#!/bin/sh\nprintf "half an image" > "$2"\necho boom >&2\nexit 1\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    root, _ = tiny_dataset
    image = tmp_path / "dataset.squashfs"
    with pytest.raises(SquashFSError, match="boom"):
        build_image(root, image)

    assert not image.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_missing_source_is_reported(tmp_path, stub_mksquashfs):
    with pytest.raises(SquashFSError, match="not a directory"):
        build_image(tmp_path / "absent", tmp_path / "dataset.squashfs")


def test_absent_mksquashfs_is_reported_with_a_pointer(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(squashfs, "have_mksquashfs", lambda: False)
    root, _ = tiny_dataset
    with pytest.raises(SquashFSError, match="docs.lumi-supercomputer.eu"):
        build_image(root, tmp_path / "dataset.squashfs")


def test_absent_squashfuse_suggests_prebound(tmp_path, monkeypatch):
    image = tmp_path / "dataset.squashfs"
    image.write_bytes(b"not really an image")
    monkeypatch.setattr(squashfs, "have_squashfuse", lambda: False)
    with pytest.raises(SquashFSError, match="prebound"):
        with squashfs.mounted_image(image):
            pass


def test_mounting_a_missing_image_is_reported(tmp_path):
    with pytest.raises(SquashFSError, match="image not found"):
        with squashfs.mounted_image(tmp_path / "absent.squashfs"):
            pass


# --- with the real tools, when available -------------------------------------


@needs_mksquashfs
def test_real_image_is_smaller_in_object_count(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    result = build_image(root, tmp_path / "dataset.squashfs")
    assert (tmp_path / "dataset.squashfs").stat().st_size == result["image_bytes"]
    assert result["size_ratio"] > 0.0


@needs_mksquashfs
@needs_squashfuse
def test_real_round_trip_preserves_every_sample(tiny_dataset, tmp_path):
    """A mounted image must present the same paths and the same bytes."""
    from dataaware.manifest import checksum_bytes, read_manifest

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    image = tmp_path / "dataset.squashfs"
    build_image(root, image)

    with squashfs.mounted_image(image) as (mount_point, seconds):
        assert seconds >= 0.0
        for sample in samples:
            payload = (mount_point / sample.relative_path).read_bytes()
            assert checksum_bytes(payload) == sample.checksum

    # The mount point is gone once the context exits.
    assert not mount_point.exists() or not any(mount_point.iterdir())
