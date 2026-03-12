import pandas as pd
import pytest
from transformers import AutoTokenizer

from bh24_literature_mining.config import LABEL2ID, LABEL_LIST
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    build_single_dataset,
    get_token_ner_tags,
    tokenize_and_align_labels,
    tokenize_dataset,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bioformers/bioformer-16L")


@pytest.fixture
def sample_iob_df():
    return pd.DataFrame(
        {
            "tokens": ["Use", "BLAST", "here", "", "Run", "it", ""],
            "ner_tags": ["O", "B-BT", "O", "", "O", "O", ""],
        }
    )


def test_get_token_ner_tags(sample_iob_df):
    tokens, tags, df = get_token_ner_tags(sample_iob_df, LABEL2ID)
    assert len(tokens) == 2
    assert len(tags) == 2
    assert len(df) == 2
    assert df.iloc[0]["tokens"] == ["Use", "BLAST", "here"]


def test_build_hf_dataset(sample_iob_df):
    _, _, df = get_token_ner_tags(sample_iob_df, LABEL2ID)
    ds = build_hf_dataset(df, df, LABEL_LIST)
    assert "train" in ds
    assert "validation" in ds
    assert len(ds["train"]) == 2


def test_build_single_dataset(sample_iob_df):
    _, _, df = get_token_ner_tags(sample_iob_df, LABEL2ID)
    ds = build_single_dataset(df, LABEL_LIST)
    assert len(ds) == 2
    assert "tokens" in ds.column_names
    assert "ner_tags" in ds.column_names


def test_tokenize_and_align_labels(tokenizer, sample_iob_df):
    _, _, df = get_token_ner_tags(sample_iob_df, LABEL2ID)
    ds = build_hf_dataset(df, df, LABEL_LIST)
    tokenized = ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    assert "labels" in tokenized["train"].column_names
    labels = tokenized["train"][0]["labels"]
    assert -100 in labels  # special tokens
    assert len(labels) == 512  # padded to max_length


def test_tokenize_dataset(tokenizer, sample_iob_df):
    _, _, df = get_token_ner_tags(sample_iob_df, LABEL2ID)
    ds = build_hf_dataset(df, df, LABEL_LIST)
    tokenized = tokenize_dataset(ds, tokenizer)
    assert "labels" in tokenized["train"].column_names
