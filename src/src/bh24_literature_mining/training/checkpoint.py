import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_checkpoint_dirs(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_dir() and p.name.startswith("checkpoint")),
        key=lambda p: p.stat().st_mtime,
    )


def get_last_created_checkpoint(directory: str | Path) -> Path | None:
    folders = get_checkpoint_dirs(directory)
    return folders[-1] if folders else None


def cleanup_checkpoints(
    output_dir: str | Path,
    best_model_dir: str | Path | None = None,
    keep_last: bool = True,
) -> None:
    checkpoints = get_checkpoint_dirs(output_dir)
    if not checkpoints:
        return

    keep: set[Path] = set()
    if best_model_dir is not None:
        keep.add(Path(best_model_dir).resolve())
    if keep_last:
        keep.add(checkpoints[-1].resolve())

    for cp in checkpoints:
        if cp.resolve() not in keep:
            logger.info("Removing checkpoint: %s", cp)
            shutil.rmtree(cp)
