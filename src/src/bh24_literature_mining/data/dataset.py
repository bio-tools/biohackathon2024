from pathlib import Path
from typing import Any

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from transformers import PreTrainedTokenizerBase


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
    val_file: str = "val_IOB.tsv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep="\t",
            names=["tokens", "ner_tags"],
            skip_blank_lines=False,
            na_filter=False,
        )

    return _read(data_dir / train_file), _read(data_dir / val_file)


def get_label_list(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["ner_tags"].dropna()) - {""})


def _ner_features(label_list: list[str]) -> Features:
    return Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )


def build_hf_dataset(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    label_list: list[str],
) -> DatasetDict:
    features = _ner_features(label_list)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, features=features),
            "validation": Dataset.from_pandas(dev_df, features=features),
        }
    )


def build_single_dataset(
    df: pd.DataFrame,
    label_list: list[str],
) -> Dataset:
    return Dataset.from_pandas(df, features=_ner_features(label_list))


def tokenize_and_align_labels(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    label_all_tokens: bool = False,
) -> dict[str, Any]:
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=512,
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx: int | None = None
        label_ids: list[int] = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    label_all_tokens: bool = False,
) -> DatasetDict:
    return ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_all_tokens),
        batched=True,
    )
