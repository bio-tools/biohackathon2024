import logging
import torch
from pathlib import Path
from typing import Callable
from datetime import datetime

from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from bh24_literature_mining.config import ID2LABEL, LABEL2ID, LABEL_LIST, PipelineConfig, TrainingConfig
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    get_token_ner_tags,
    load_iob_splits,
    tokenize_and_align_labels,
)
from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.evaluation import compute_metrics
from bh24_literature_mining.models import create_model
from bh24_literature_mining.training.checkpoint import cleanup_checkpoints

logger = logging.getLogger(__name__)


def build_training_args(
    config: TrainingConfig, project_root: Path
) -> TrainingArguments:
    now = datetime.now()
    return TrainingArguments(
        run_name=now.strftime("%y%m%d-%H:%M"),
        output_dir=str(project_root / config.output_dir),
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        label_smoothing_factor = config.label_smoothing_factor,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
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
    config = TrainingConfig,
    
) -> Trainer:
    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )
        logger.info("Early stopping enabled with patience=%d", config.early_stopping_patience)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )


def run_training(
    config: PipelineConfig,
    project_root: Path,
    resume_from: Path | None = None,
) -> dict[str, float]:
    data_dir = project_root / config.data.iob_dir
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

    model = create_model(config.model, ID2LABEL, LABEL2ID)
    training_args = build_training_args(config.training, project_root)

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics_fn=lambda p: compute_metrics(p, ID2LABEL),
        config=config.training,
    )

    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    eval_results = trainer.evaluate()

    output_dir = project_root / config.training.output_dir
    best_model_dir = getattr(trainer.state, "best_model_checkpoint", None)
    cleanup_checkpoints(output_dir, best_model_dir=best_model_dir, keep_last=True)

    return eval_results
