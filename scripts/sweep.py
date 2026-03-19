"""W&B hyperparameter sweep for NER model."""

import logging
from pathlib import Path

import wandb
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from bh24_literature_mining.config import ID2LABEL, LABEL2ID, LABEL_LIST, load_config
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    get_token_ner_tags,
    load_iob_splits,
    tokenize_and_align_labels,
)
from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.evaluation import compute_metrics
from bh24_literature_mining.models.ner_model import create_model
from bh24_literature_mining.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "no_aug.yaml"

sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval/f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 3.5e-5,
            "max": 6.5e-5,
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.17,
            "max": 0.24,
        },
        "warmup_ratio": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.18,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 0.008,
            "max": 0.025,
        },
        "label_smoothing_factor": {"values": [0.15, 0.2, 0.25]},
        "batch_size": {"value": 16},
        "gradient_accumulation_steps": {"value": 2},
    },
}


def train() -> None:
    run = wandb.init()
    wconfig = wandb.config

    config = load_config(CONFIG_PATH)

    data_dir = PROJECT_ROOT / config.data.iob_dir
    train_raw, val_raw = load_iob_splits(
        data_dir, train_file="train_IOB.tsv", val_file="val_IOB.tsv"
    )
    _, _, train_df = get_token_ner_tags(train_raw, LABEL2ID)
    _, _, val_df = get_token_ner_tags(val_raw, LABEL2ID)

    tokenizer = get_tokenizer(config.model.pretrained)
    ds = build_hf_dataset(train_df, val_df, LABEL_LIST)
    tokenized_ds = ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    model_config = ModelConfig(
        pretrained=config.model.pretrained,
        dropout=wconfig.dropout,
        num_labels=config.model.num_labels,
    )
    model = create_model(model_config, ID2LABEL, LABEL2ID)

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "sweep" / run.name),
        run_name=run.name,
        learning_rate=wconfig.learning_rate,
        per_device_train_batch_size=wconfig.batch_size,
        per_device_eval_batch_size=wconfig.batch_size,
        gradient_accumulation_steps=wconfig.gradient_accumulation_steps,
        num_train_epochs=15,
        warmup_ratio=wconfig.warmup_ratio,
        weight_decay=wconfig.weight_decay,
        label_smoothing_factor=wconfig.label_smoothing_factor,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=str(PROJECT_ROOT / "logs" / "sweep" / run.name),
        logging_steps=250,
        optim="adamw_torch",
        bf16=False,
        seed=42,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, ID2LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    wandb.log({"best_f1": eval_results["eval_f1"]})
    logger.info("Run %s — F1: %.4f", run.name, eval_results["eval_f1"])
    run.finish()


def main() -> None:
    sweep_id = wandb.sweep(sweep_config, project="biohackathon-ner-sweep")
    logger.info("Sweep ID: %s", sweep_id)
    wandb.agent(sweep_id, function=train, count=20)


if __name__ == "__main__":
    main()
