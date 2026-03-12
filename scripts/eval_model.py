import argparse
import logging
import tempfile
from pathlib import Path

from transformers import BertForTokenClassification, Trainer, TrainingArguments

from bh24_literature_mining.config import ID2LABEL, LABEL2ID, LABEL_LIST, load_config
from bh24_literature_mining.data.dataset import (
    build_single_dataset,
    get_token_ner_tags,
    load_iob_splits,
    tokenize_and_align_labels,
)
from bh24_literature_mining.data.tokenizer import get_tokenizer
from bh24_literature_mining.evaluation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NER model checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    config = load_config(args.config)

    data_dir = args.test_data or (project_root / config.data.iob_dir)
    _, test_raw = load_iob_splits(data_dir, val_file="test_IOB.tsv")
    _, _, test_df = get_token_ner_tags(test_raw, LABEL2ID)

    tokenizer = get_tokenizer(config.model.pretrained)

    test_ds = build_single_dataset(test_df, LABEL_LIST)
    tokenized_test = test_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    model = BertForTokenClassification.from_pretrained(str(args.checkpoint))

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_eval_batch_size=config.training.batch_size,
            seed=config.training.seed,
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, ID2LABEL),
        )

        eval_results = trainer.evaluate()

    logger.info("F1: %.4f", eval_results["eval_f1"])
    logger.info("Precision: %.4f", eval_results["eval_precision"])
    logger.info("Recall: %.4f", eval_results["eval_recall"])
    logger.info("Accuracy: %.4f", eval_results["eval_accuracy"])


if __name__ == "__main__":
    main()
