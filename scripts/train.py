import argparse
import logging
from pathlib import Path

from bh24_literature_mining.config import ID2LABEL, LABEL2ID, LABEL_LIST, load_config
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    get_token_ner_tags,
    load_iob_splits,
    tokenize_and_align_labels,
)
from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.evaluation import compute_metrics
from bh24_literature_mining.models import create_model
from bh24_literature_mining.training import build_training_args, cleanup_checkpoints
from bh24_literature_mining.training.trainer import build_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    config = load_config(args.config)

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
        early_stopping_patience=config.training.early_stopping_patience,
    )

    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    eval_results = trainer.evaluate()

    logger.info("F1: %.4f", eval_results["eval_f1"])
    logger.info("Precision: %.4f", eval_results["eval_precision"])
    logger.info("Recall: %.4f", eval_results["eval_recall"])
    logger.info("Accuracy: %.4f", eval_results["eval_accuracy"])

    output_dir = project_root / config.training.output_dir
    best_model_dir = getattr(trainer.state, "best_model_checkpoint", None)
    cleanup_checkpoints(output_dir, best_model_dir=best_model_dir, keep_last=True)


if __name__ == "__main__":
    main()
