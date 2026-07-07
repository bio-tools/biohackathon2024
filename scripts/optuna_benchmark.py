"""Optuna benchmark for token-classification NER checkpoints on IOB_260507."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from bh24_literature_mining.config import ID2LABEL, LABEL2ID, LABEL_LIST
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    build_single_dataset,
    get_token_ner_tags,
    load_iob_splits,
    tokenize_and_align_labels,
)
from bh24_literature_mining.evaluation import compute_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "IOB_260507"
BIOFORMER_BASELINE = PROJECT_ROOT / "models" / "sweep" / "260616-09:29" / "checkpoint-2500"

DEFAULT_MODELS = ["bert", "biobert", "scibert", "pubmedbert", "deberta", "modernbert"]
MODEL_IDS = {
    "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "deberta": "microsoft/deberta-v3-base",
    "modernbert": "answerdotai/ModernBERT-base",
}
METRIC_FIELDS = ["f1", "precision", "recall", "accuracy", "roc_auc", "loss"]


@dataclass(frozen=True)
class TrialParams:
    learning_rate: float
    weight_decay: float
    dropout: float
    warmup_ratio: float
    label_smoothing_factor: float
    batch_size: int
    gradient_accumulation_steps: int


class OptunaPruningCallback(TrainerCallback):
    """Report validation F1 to Optuna and prune underperforming trials."""

    def __init__(self, trial: Any) -> None:
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        if not metrics or "eval_f1" not in metrics:
            return control
        step = state.global_step if state.global_step is not None else 0
        self.trial.report(metrics["eval_f1"], step=step)
        if self.trial.should_prune():
            import optuna

            raise optuna.TrialPruned(f"Pruned at step {step}")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[42, 123, 2024, 3407, 777])
    parser.add_argument("--study-dir", type=Path, default=PROJECT_ROOT / "results" / "optuna_260507")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is required for benchmark execution. Install dependencies with "
            "`poetry install` or `poetry add optuna sentencepiece`."
        ) from exc
    return optuna


def resolve_model_ref(model_ref: str | Path) -> str:
    path = Path(model_ref)
    if path.is_absolute() and path.exists():
        return str(path)
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return str(project_path)
    return str(model_ref)


def load_tokenized_datasets(model_id: str) -> tuple[Any, DatasetDict]:
    train_raw, val_raw = load_iob_splits(DATA_DIR, "train_IOB.tsv", "val_IOB.tsv")
    test_raw, _ = load_iob_splits(DATA_DIR, "test_IOB.tsv", "test_IOB.tsv")
    _, _, train_df = get_token_ner_tags(train_raw, LABEL2ID)
    _, _, val_df = get_token_ner_tags(val_raw, LABEL2ID)
    _, _, test_df = get_token_ner_tags(test_raw, LABEL2ID)

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_ref(model_id), use_fast=True)
    ds = build_hf_dataset(train_df, val_df, LABEL_LIST)
    ds["test"] = build_single_dataset(test_df, LABEL_LIST)
    tokenized = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    return tokenizer, tokenized


def apply_dropout(config: Any, dropout: float) -> Any:
    for name in (
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "hidden_dropout",
        "attention_dropout",
        "classifier_dropout",
        "classifier_dropout_prob",
    ):
        if hasattr(config, name):
            setattr(config, name, dropout)
    return config


def create_model(model_id: str, dropout: float) -> AutoModelForTokenClassification:
    model_ref = resolve_model_ref(model_id)
    config = AutoConfig.from_pretrained(
        model_ref,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    config = apply_dropout(config, dropout)
    return AutoModelForTokenClassification.from_pretrained(
        model_ref,
        config=config,
        ignore_mismatched_sizes=True,
    )


def training_args(
    output_dir: Path,
    logging_dir: Path,
    params: TrialParams,
    seed: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        num_train_epochs=15,
        warmup_ratio=params.warmup_ratio,
        weight_decay=params.weight_decay,
        label_smoothing_factor=params.label_smoothing_factor,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=str(logging_dir),
        logging_steps=500,
        optim="adamw_torch",
        bf16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        report_to="none",
    )


def metrics_without_prefix(metrics: dict[str, Any], prefix: str = "eval_") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith(prefix):
            out[key.removeprefix(prefix)] = value
    return out


def sampled_params(trial: Any) -> TrialParams:
    return TrialParams(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        dropout=trial.suggest_float("dropout", 0.05, 0.35),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        label_smoothing_factor=trial.suggest_float("label_smoothing_factor", 0.0, 0.2),
        batch_size=trial.suggest_categorical("batch_size", [8, 16]),
        gradient_accumulation_steps=trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
    )


def run_trainer(
    model_id: str,
    tokenizer: Any,
    tokenized: DatasetDict,
    params: TrialParams,
    output_dir: Path,
    logging_dir: Path,
    seed: int,
    trial: Any | None = None,
) -> tuple[Trainer, dict[str, Any]]:
    set_seed(seed)
    trainer = Trainer(
        model=create_model(model_id, params.dropout),
        args=training_args(output_dir, logging_dir, params, seed),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, ID2LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        + ([OptunaPruningCallback(trial)] if trial is not None else []),
    )
    trainer.train()
    metrics = trainer.evaluate()
    return trainer, metrics


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_non_best_trials(model_dir: Path, best_trial_number: int | None) -> None:
    if best_trial_number is None or not model_dir.exists():
        return
    keep = f"trial-{best_trial_number}"
    for path in model_dir.glob("trial-*"):
        if path.is_dir() and path.name != keep:
            shutil.rmtree(path)


def run_study(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    optuna = require_optuna()
    model_id = MODEL_IDS[model_key]
    tokenizer, tokenized = load_tokenized_datasets(model_id)
    model_trial_dir = PROJECT_ROOT / "models" / "optuna_260507" / model_key
    log_dir = PROJECT_ROOT / "logs" / "optuna_260507" / model_key
    args.study_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{args.study_dir / 'optuna.db'}"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"260507_{model_key}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def objective(trial: Any) -> float:
        params = sampled_params(trial)
        trial_dir = model_trial_dir / f"trial-{trial.number}"
        trainer, metrics = run_trainer(
            model_id=model_id,
            tokenizer=tokenizer,
            tokenized=tokenized,
            params=params,
            output_dir=trial_dir,
            logging_dir=log_dir / f"trial-{trial.number}",
            seed=42,
            trial=trial,
        )
        trial.set_user_attr("params", asdict(params))
        trial.set_user_attr("metrics", metrics_without_prefix(metrics))
        trial.set_user_attr("best_checkpoint", trainer.state.best_model_checkpoint)
        return float(metrics["eval_f1"])

    remaining = max(args.trials - len(study.trials), 0)
    if remaining:
        logger.info("Running %s Optuna trials for %s", remaining, model_key)
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("Study %s already has at least %s trials", study.study_name, args.trials)

    best = study.best_trial
    cleanup_non_best_trials(model_trial_dir, best.number)
    payload = {
        "model_key": model_key,
        "model_id": model_id,
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }
    write_json(args.study_dir / f"{model_key}_best.json", payload)
    return payload


def evaluate_checkpoint(
    checkpoint: Path,
    tokenizer_id: str | Path,
    tokenized: DatasetDict,
    split: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    trainer = Trainer(
        model=AutoModelForTokenClassification.from_pretrained(checkpoint),
        args=TrainingArguments(
            output_dir=str(PROJECT_ROOT / "models" / "tmp_eval"),
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=torch.cuda.is_available(),
        ),
        eval_dataset=tokenized[split],
        processing_class=AutoTokenizer.from_pretrained(resolve_model_ref(tokenizer_id), use_fast=True),
        compute_metrics=lambda p: compute_metrics(p, ID2LABEL),
    )
    prefix = "test" if split == "test" else "eval"
    return metrics_without_prefix(trainer.evaluate(metric_key_prefix=prefix), prefix=f"{prefix}_")


def run_final_seeds(
    model_key: str,
    best_params: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    model_id = MODEL_IDS[model_key]
    tokenizer, tokenized = load_tokenized_datasets(model_id)
    params = TrialParams(**best_params)
    rows = []
    for seed in args.final_seeds:
        logger.info("Final retrain for %s seed %s", model_key, seed)
        output_dir = PROJECT_ROOT / "models" / "benchmark_260507" / model_key / f"seed-{seed}"
        log_dir = PROJECT_ROOT / "logs" / "benchmark_260507" / model_key / f"seed-{seed}"
        trainer, metrics = run_trainer(
            model_id=model_id,
            tokenizer=tokenizer,
            tokenized=tokenized,
            params=params,
            output_dir=output_dir,
            logging_dir=log_dir,
            seed=seed,
        )
        row = {
            "model_key": model_key,
            "model_id": model_id,
            "seed": seed,
            "checkpoint": trainer.state.best_model_checkpoint or str(output_dir),
        }
        row.update(metrics_without_prefix(metrics))
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model_key in sorted({row["model_key"] for row in rows}):
        model_rows = [row for row in rows if row["model_key"] == model_key]
        agg: dict[str, Any] = {"model_key": model_key, "n": len(model_rows)}
        for metric in METRIC_FIELDS:
            values = [row.get(metric) for row in model_rows]
            values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
            if values:
                agg[f"{metric}_mean"] = mean(values)
                agg[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(agg)
    return sorted(out, key=lambda row: row.get("f1_mean", float("-inf")), reverse=True)


def add_bioformer_baseline(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if not BIOFORMER_BASELINE.exists():
        logger.warning("Bioformer baseline checkpoint not found: %s", BIOFORMER_BASELINE)
        return
    tokenizer, tokenized = load_tokenized_datasets(str(BIOFORMER_BASELINE))
    metrics = evaluate_checkpoint(BIOFORMER_BASELINE, BIOFORMER_BASELINE, tokenized, "validation")
    row = {
        "model_key": "bioformer_baseline",
        "model_id": str(BIOFORMER_BASELINE.relative_to(PROJECT_ROOT)),
        "seed": "baseline",
        "checkpoint": str(BIOFORMER_BASELINE),
    }
    row.update(metrics)
    rows.append(row)


def evaluate_best_on_test(args: argparse.Namespace, final_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_rows = [row for row in final_rows if row.get("f1") is not None and row.get("checkpoint")]
    if not candidate_rows:
        return None
    best = max(candidate_rows, key=lambda row: row.get("f1", float("-inf")))
    tokenizer, tokenized = load_tokenized_datasets(best["model_id"])
    metrics = evaluate_checkpoint(Path(best["checkpoint"]), best["model_id"], tokenized, "test")
    payload = {
        "selected_by": "best_final_validation_f1",
        "model_key": best["model_key"],
        "model_id": best["model_id"],
        "seed": best["seed"],
        "checkpoint": best["checkpoint"],
        "validation_f1": best.get("f1"),
        "test_metrics": metrics,
    }
    write_json(args.study_dir / "single_best_test_eval.json", payload)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {
        "models": {key: MODEL_IDS[key] for key in args.models},
        "trials_per_model": args.trials,
        "final_seeds": args.final_seeds,
        "dataset": str(DATA_DIR.relative_to(PROJECT_ROOT)),
        "study_dir": str(args.study_dir),
        "study_storage": str(args.study_dir / "optuna.db"),
        "trial_checkpoints": "models/optuna_260507/<model_key>/trial-<n>/",
        "final_checkpoints": "models/benchmark_260507/<model_key>/seed-<seed>/",
        "bioformer_baseline": str(BIOFORMER_BASELINE.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.study_dir = args.study_dir if args.study_dir.is_absolute() else PROJECT_ROOT / args.study_dir
    if args.dry_run:
        dry_run(args)
        return

    study_payloads = [run_study(model_key, args) for model_key in args.models]
    write_json(args.study_dir / "study_summary.json", study_payloads)
    write_csv(
        args.study_dir / "study_summary.csv",
        [
            {
                "model_key": item["model_key"],
                "model_id": item["model_id"],
                "best_trial": item["best_trial"],
                "best_value": item["best_value"],
                "n_trials": item["n_trials"],
                **item["best_params"],
            }
            for item in study_payloads
        ],
    )

    final_rows: list[dict[str, Any]] = []
    for item in study_payloads:
        final_rows.extend(run_final_seeds(item["model_key"], item["best_params"], args))
    add_bioformer_baseline(args, final_rows)

    aggregate = aggregate_rows(final_rows)
    test_payload = evaluate_best_on_test(args, final_rows)
    write_csv(args.study_dir / "final_seed_results.csv", final_rows)
    write_csv(args.study_dir / "final_aggregate_results.csv", aggregate)
    write_json(
        args.study_dir / "final_results.json",
        {"rows": final_rows, "aggregate": aggregate, "single_best_test_eval": test_payload},
    )


if __name__ == "__main__":
    main()
