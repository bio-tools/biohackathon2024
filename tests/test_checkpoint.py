import os
import tempfile
import time
from pathlib import Path

from bh24_literature_mining.training.checkpoint import (
    cleanup_checkpoints,
    get_checkpoint_dirs,
    get_last_created_checkpoint,
)


def _make_checkpoints(base: Path, names: list[str]) -> list[Path]:
    dirs = []
    for name in names:
        d = base / name
        d.mkdir()
        (d / "model.bin").touch()
        dirs.append(d)
        time.sleep(0.01)
    return dirs


def test_get_checkpoint_dirs_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _make_checkpoints(base, ["checkpoint-100", "checkpoint-50", "checkpoint-200"])
        dirs = get_checkpoint_dirs(base)
        names = [d.name for d in dirs]
        assert names[0] == "checkpoint-100" or names[0] == "checkpoint-50"
        assert len(dirs) == 3


def test_get_checkpoint_dirs_ignores_non_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "other_dir").mkdir()
        _make_checkpoints(base, ["checkpoint-1"])
        dirs = get_checkpoint_dirs(base)
        assert len(dirs) == 1


def test_get_last_created_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _make_checkpoints(base, ["checkpoint-1", "checkpoint-2", "checkpoint-3"])
        last = get_last_created_checkpoint(base)
        assert last is not None
        assert last.name == "checkpoint-3"


def test_get_last_created_checkpoint_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert get_last_created_checkpoint(tmp) is None


def test_cleanup_keeps_best_and_last():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        cps = _make_checkpoints(
            base, ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
        )
        best = cps[1]
        cleanup_checkpoints(base, best_model_dir=best, keep_last=True)
        remaining = get_checkpoint_dirs(base)
        remaining_names = {d.name for d in remaining}
        assert "checkpoint-2" in remaining_names  # best
        assert "checkpoint-3" in remaining_names  # last
        assert "checkpoint-1" not in remaining_names


def test_cleanup_keeps_only_best():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        cps = _make_checkpoints(base, ["checkpoint-1", "checkpoint-2", "checkpoint-3"])
        best = cps[0]
        cleanup_checkpoints(base, best_model_dir=best, keep_last=False)
        remaining = get_checkpoint_dirs(base)
        assert len(remaining) == 1
        assert remaining[0].name == "checkpoint-1"


def test_cleanup_no_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        cleanup_checkpoints(tmp)
