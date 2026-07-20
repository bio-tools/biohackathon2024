"""Prepare negative-inclusive, PMCID- and tool-disjoint IOB splits."""

from pathlib import Path

from bh24_literature_mining.data.preparation import prepare_iob_splits

CSV = Path("data/annotated/260507_annotated.csv")
OUT = Path("data/IOB_260713")
MODEL = "bioformers/bioformer-16L"
SEED = 42


def main() -> None:
    summary = prepare_iob_splits(
        annotations_path=CSV,
        output_dir=OUT,
        tokenizer_name=MODEL,
        random_seed=SEED,
    )
    print(summary)
    print(f"Saved IOB files to {OUT}")


if __name__ == "__main__":
    main()
