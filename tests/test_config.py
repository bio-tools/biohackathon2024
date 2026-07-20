import tempfile
from pathlib import Path

import yaml

from bh24_literature_mining.config import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    NUM_LABELS,
    PipelineConfig,
    load_config,
)


def test_label_constants_consistent():
    assert len(LABEL_LIST) == NUM_LABELS
    assert all(ID2LABEL[LABEL2ID[lbl]] == lbl for lbl in LABEL_LIST)
    assert all(LABEL2ID[ID2LABEL[i]] == i for i in range(NUM_LABELS))


def test_label_list_contents():
    assert "O" in LABEL_LIST
    assert "B-BT" in LABEL_LIST
    assert "I-BT" in LABEL_LIST


def test_load_config_defaults():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        f.flush()
        config = load_config(Path(f.name))
    assert config.training.epochs == 15
    assert config.model.pretrained == "bioformers/bioformer-16L"
    assert config.data.random_seed == 42


def test_load_config_overrides():
    raw = {"training": {"epochs": 10, "learning_rate": 2e-5}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        f.flush()
        config = load_config(Path(f.name))
    assert config.training.epochs == 10
    assert config.training.learning_rate == 2e-5
    assert config.training.batch_size == 16


def test_load_config_path_fields():
    raw = {"training": {"output_dir": "custom/path"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        f.flush()
        config = load_config(Path(f.name))
    assert isinstance(config.training.output_dir, Path)
    assert config.training.output_dir == Path("custom/path")
