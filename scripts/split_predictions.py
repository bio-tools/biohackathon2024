"""Split predicted CSV into exact / extended / negative subsets."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "annotated" / "merged_annotated_predicted.csv"
OUT_EXACT = ROOT / "data" / "annotated" / "merged_exact.csv"
OUT_EXTENDED = ROOT / "data" / "annotated" / "merged_extended.csv"
OUT_NEGATIVE = ROOT / "data" / "annotated" / "merged_negative.csv"


def count_spans(tag_str: str) -> int:
    if not isinstance(tag_str, str) or tag_str.strip() == "":
        return 0
    return len(re.findall(r"\(\d+", tag_str))


def main() -> None:
    df = pd.read_csv(INPUT)
    df["NER_Tags"] = df["NER_Tags"].fillna("")
    df["NER_tags_predicted"] = df["NER_tags_predicted"].fillna("")

    no_pred = df["NER_tags_predicted"] == ""
    exact = (~no_pred) & (df["NER_tags_predicted"] == df["NER_Tags"])
    extended = (~no_pred) & (~exact)

    df[exact].to_csv(OUT_EXACT, index=False)
    df[extended].to_csv(OUT_EXTENDED, index=False)
    df[no_pred].to_csv(OUT_NEGATIVE, index=False)

    print(f"exact:    {exact.sum():>5}  → {OUT_EXACT}")
    print(f"extended: {extended.sum():>5}  → {OUT_EXTENDED}")
    print(f"negative: {no_pred.sum():>5}  → {OUT_NEGATIVE}")


if __name__ == "__main__":
    main()
