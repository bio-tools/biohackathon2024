import pandas as pd
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    annotated_path = root / "data/annotated/260226_annotated.csv"
    new_path = root / "data/annotated/260302_mentions_with_topics_clean.csv"
    output_path = root / "data/annotated/merged_annotated.csv"

    cols = ["PMCID", "Sentence", "True?", "False?", "NER_Tags"]

    df1 = pd.read_csv(annotated_path)[cols]
    df2 = pd.read_csv(new_path)
    df2 = df2[df2["True?"] == True][cols]

    merged = pd.concat([df1, df2], ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
