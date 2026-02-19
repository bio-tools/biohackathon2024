import os
import shutil
from pathlib import Path


def cleanup_checkpoints(
    output_dir: str,
    keep_last: bool = True,
    best_model_dir: str | None = None,
    last_model_dir: str | None = None,
) -> None:
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            if item_path != best_model_dir and (
                not keep_last or item_path != last_model_dir
            ):
                shutil.rmtree(item_path)


def get_last_created_checkpoint(directory: str | Path) -> Path | None:
    folders = [
        d
        for d in Path(directory).iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    return max(folders, key=os.path.getctime) if folders else None
