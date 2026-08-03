"""Shared fixtures.

Adds the repository root to ``sys.path`` so the tests run against the working
tree without an install step, which is how they run on a login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataaware.generate import PROFILES, generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate a very small dataset once per test session.

    Returns ``(dataset_root, manifest_path)``. Smaller than the ``tiny`` profile:
    tests should be fast, and none of them measure performance.
    """
    from dataclasses import replace

    root = tmp_path_factory.mktemp("tiny-dataset")
    profile = replace(PROFILES["tiny"], samples=32, classes=4)
    manifest = generate_dataset(profile, root, overwrite=True)
    return root, manifest


@pytest.fixture
def base_config_dict(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> dict:
    """A valid configuration pointing at the session's tiny dataset."""
    root, manifest = tiny_dataset
    return {
        "run": {"name": "unit-test", "seed": 7, "warmup_batches": 0, "measured_batches": 2},
        "dataset": {
            "layout": "loose-files",
            "root": str(root),
            "manifest": str(manifest),
        },
        "loader": {
            "batch_size": 8,
            "num_workers": 0,
            "persistent_workers": False,
            "shuffle": True,
            "drop_last": True,
            "compute_steps": 1,
        },
        "distributed": {"enabled": False},
        "storage": {"location": "local"},
        "output": {"directory": str(tmp_path / "out")},
    }
