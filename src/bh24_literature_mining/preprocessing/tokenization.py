from typing import Any

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase


def tokenize_and_align_labels(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    label_all_tokens: bool = True,
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
    label_all_tokens: bool = True,
) -> DatasetDict:
    return ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_all_tokens),
        batched=True,
    )
