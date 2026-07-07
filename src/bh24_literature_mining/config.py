from dataclasses import dataclass, field
from pathlib import Path

import yaml

LABEL_LIST: list[str] = ["B-BT", "I-BT", "O"]
ID2LABEL: dict[int, str] = dict(enumerate(LABEL_LIST))
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
NUM_LABELS: int = len(LABEL_LIST)


@dataclass
class DataConfig:
    annotations_path: Path = Path("data/annotated/251105_annotated.csv")
    iob_dir: Path = Path("data/IOB")
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    augment: bool = False
    augment_n_copies: int = 1
    biotools_path: Path = Path("biotoolspub/biotoolspub_with_topic.tsv")
    include_negatives: bool = True


@dataclass
class ModelConfig:
    pretrained: str = "bioformers/bioformer-16L"
    dropout: float = 0.1
    num_labels: int = NUM_LABELS


@dataclass
class TrainingConfig:
    epochs: int = 15
    max_steps: int = -1
    learning_rate: float = 1e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.2
    bf16: bool = True
    output_dir: Path = Path("models")
    save_strategy: str = "steps"
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    seed: int = 42
    logging_dir: Path = Path("logs")
    logging_steps: int = 500
    early_stopping_patience: int = 3
    label_smoothing_factor: float = 0.1


@dataclass
class InferenceConfig:
    checkpoint_path: Path = Path("models/checkpoint-14050")
    input_path: Path = Path("data/to_predict/250805_mentions_with_topics.csv")
    output_path: Path = Path("data/predicted/results.csv")
    batch_size: int = 16
    max_length: int = 512


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _set_nested(obj: object, keys: list[str], value: object) -> None:
    for key in keys[:-1]:
        obj = getattr(obj, key)

    field = keys[-1]
    current = getattr(obj, field)
    target_type = type(current)

    if target_type is Path:
        value = Path(value)
    elif target_type is float:
        value = float(value)
    elif target_type is int:
        value = int(value)
    elif target_type is bool:
        value = bool(value)

    setattr(obj, field, value)


def load_config(path: Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = PipelineConfig()
    for section, values in (raw or {}).items():
        if not isinstance(values, dict):
            continue
        for key, val in values.items():
            _set_nested(config, [section, key], val)
    return config
