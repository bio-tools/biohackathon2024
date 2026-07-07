import argparse
import logging
from pathlib import Path

from bh24_literature_mining.config import load_config
from bh24_literature_mining.training import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    config = load_config(args.config)

    eval_results = run_training(config, project_root, resume_from=args.resume_from)

    logger.info("F1: %.4f", eval_results["eval_f1"])
    logger.info("Precision: %.4f", eval_results["eval_precision"])
    logger.info("Recall: %.4f", eval_results["eval_recall"])
    logger.info("Accuracy: %.4f", eval_results["eval_accuracy"])


if __name__ == "__main__":
    main()
