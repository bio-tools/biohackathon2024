"""Report whether strict PMCID- and tool-disjoint splitting is feasible."""

from pathlib import Path

from bh24_literature_mining.data.annotation_parser import prepare_annotations
from bh24_literature_mining.data.preparation import count_negative_rows
from bh24_literature_mining.data.splitter import (
    get_dataframe_resource_ids,
    get_document_tool_components,
    split_by_pmcid_and_resource,
)

CSV = Path("data/annotated/260507_annotated.csv")


def main() -> None:
    df = prepare_annotations(CSV, entity_type=None, include_negatives=True)
    components = get_document_tool_components(df)
    sentence_counts = {
        str(pmcid): int(count)
        for pmcid, count in df.groupby("PMCID").size().items()
    }
    print(f"sentences={len(df)}")
    print(f"documents={df['PMCID'].nunique()}")
    print(f"components={len(components)}")
    for index, component in enumerate(components[:20], start=1):
        sentences = sum(sentence_counts[pmcid] for pmcid in component)
        print(
            f"component={index} documents={len(component)} "
            f"sentences={sentences}"
        )
    train, validation, test = split_by_pmcid_and_resource(df, random_seed=42)
    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        print(
            f"split={name} sentences={len(split)} "
            f"negatives={count_negative_rows(split)} "
            f"resource_ids={len(get_dataframe_resource_ids(split))}"
        )


if __name__ == "__main__":
    main()
