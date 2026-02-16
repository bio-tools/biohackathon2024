from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value


def convert_IOB_transformer(flat_list: list, pattern: object) -> list[list]:
    new_list: list[list] = []
    sub_list: list = []
    for item in flat_list:
        if item != pattern:
            sub_list.append(item)
        else:
            new_list.append(sub_list)
            sub_list = []
    return new_list


def get_token_ner_tags(
    df: pd.DataFrame, label2id: dict
) -> tuple[list[list[str]], list[list], pd.DataFrame]:
    ner_tag_list_flat = df["ner_tags"].map(label2id).fillna("###").tolist()
    token_list_flat = df["tokens"].tolist()
    token_list = convert_IOB_transformer(token_list_flat, pattern="")
    ner_tag_list = convert_IOB_transformer(ner_tag_list_flat, pattern="###")
    out_df = pd.DataFrame({"tokens": token_list, "ner_tags": ner_tag_list})
    return token_list, ner_tag_list, out_df


def load_iob_splits(
    data_dir: Path,
    train_file: str = "train_IOB.tsv",
    dev_file: str = "dev_IOB.tsv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep="\t",
            names=["tokens", "ner_tags"],
            skip_blank_lines=False,
            na_filter=False,
        )

    return _read(data_dir / train_file), _read(data_dir / dev_file)


def get_label_list(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["ner_tags"].dropna()) - {""})


def build_hf_dataset(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    label_list: list[str],
) -> DatasetDict:
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, features=features),
            "validation": Dataset.from_pandas(dev_df, features=features),
        }
    )
