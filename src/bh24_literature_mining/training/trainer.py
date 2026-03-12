import logging
from pathlib import Path
from typing import Callable

import torch
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from bh24_literature_mining.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_training_args(
    config: TrainingConfig, project_root: Path
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(project_root / config.output_dir),
        eval_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        logging_dir=str(project_root / config.logging_dir),
        logging_steps=config.logging_steps,
        bf16=config.bf16 and torch.cuda.is_available(),
        seed=config.seed,
    )


def build_trainer(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    train_dataset: object,
    eval_dataset: object,
    tokenizer: PreTrainedTokenizerBase,
    compute_metrics_fn: Callable,
    early_stopping_patience: int = 5,
) -> Trainer:
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info("Early stopping enabled with patience=%d", early_stopping_patience)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
