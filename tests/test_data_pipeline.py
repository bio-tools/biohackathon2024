import ast
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from transformers import AutoTokenizer

from bh24_literature_mining.data.annotation_parser import (
    filter_checked,
    normalize_entity_type,
    parse_ner_tags,
    prepare_annotations,
)
from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    convert_IOB_transformer,
    get_label_list,
    get_token_ner_tags,
    load_iob_splits,
)
from bh24_literature_mining.data.integrity import check_data_integrity
from bh24_literature_mining.data.iob_converter import (
    convert_to_IOB_format_from_df,
    convert_to_iob,
    find_sub_span,
    load_iob_file,
)
from bh24_literature_mining.data.splitter import split_by_pmcid


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bioformers/bioformer-16L")


# --- find_sub_span ---

def test_find_sub_span_overlap():
    assert find_sub_span((0, 5), (3, 8)) == (3, 5)


def test_find_sub_span_contained():
    assert find_sub_span((2, 7), (0, 10)) == (2, 7)


def test_find_sub_span_no_overlap():
    assert find_sub_span((0, 3), (5, 10)) is None


def test_find_sub_span_adjacent():
    assert find_sub_span((0, 3), (3, 6)) is None


# --- convert_to_iob ---

def test_convert_to_iob_no_entities(tokenizer):
    result = convert_to_iob(["hello world"], [None], tokenizer)
    assert result == [[("hello", "O"), ("world", "O")]]


def test_convert_to_iob_single_entity(tokenizer):
    text = "Use BLAST for search"
    ner_tags = [[4, 9, "BLAST", "BT"]]
    result = convert_to_iob([text], [ner_tags], tokenizer)
    tokens_tags = dict(result[0])
    assert tokens_tags["BLAST"] == "B-BT"


def test_convert_to_iob_multitoken_entity(tokenizer):
    text = "Use Galaxy Tool for analysis"
    ner_tags = [[4, 15, "Galaxy Tool", "BT"]]
    result = convert_to_iob([text], [ner_tags], tokenizer)
    tags = [tag for _, tag in result[0]]
    assert "B-BT" in tags
    assert "I-BT" in tags


def test_convert_to_iob_empty_text(tokenizer):
    result = convert_to_iob([""], [None], tokenizer)
    assert result == [[]]


def test_convert_to_iob_hyphenated_entity(tokenizer):
    text = "Use BiG-SCAPE for analysis"
    ner_tags = [[4, 13, "BiG-SCAPE", "BT"]]
    result = convert_to_iob([text], [ner_tags], tokenizer)
    tags = [tag for _, tag in result[0]]
    b_count = tags.count("B-BT")
    i_count = tags.count("I-BT")
    assert b_count == 1
    assert b_count + i_count >= 1
    first_b = next(i for i, t in enumerate(tags) if t == "B-BT")
    for j in range(first_b + 1, first_b + b_count + i_count):
        assert tags[j] == "I-BT"


# --- convert_to_IOB_format_from_df and load_iob_file ---

def test_iob_roundtrip(tokenizer):
    df = pd.DataFrame(
        {
            "Sentence": ["Use BLAST here", "Run MaxQuant now"],
            "NER_Tags": [
                [[4, 9, "BLAST", "BT"]],
                [[4, 12, "MaxQuant", "BT"]],
            ],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        convert_to_IOB_format_from_df(df, out, "test.tsv", tokenizer)
        loaded = load_iob_file(out / "test.tsv")
        tags = loaded[loaded["tag"] != ""]["tag"].tolist()
        assert "B-BT" in tags


# --- check_data_integrity ---

def test_integrity_valid_bio():
    df = pd.DataFrame(
        {"token": ["Use", "BLAST", "here"], "tag": ["O", "B-BT", "O"]}
    )
    valid, issues = check_data_integrity(df)
    assert valid
    assert issues == []


def test_integrity_valid_multi_token():
    df = pd.DataFrame(
        {"token": ["Run", "Max", "Quant"], "tag": ["O", "B-BT", "I-BT"]}
    )
    valid, issues = check_data_integrity(df)
    assert valid


def test_integrity_hanging_i_tag():
    df = pd.DataFrame(
        {"token": ["I-BT", "token"], "tag": ["I-BT", "O"]}
    )
    valid, issues = check_data_integrity(df)
    assert not valid
    assert any("Hanging" in issue for issue in issues)


def test_integrity_mismatch_b_i():
    df = pd.DataFrame(
        {"token": ["A", "B"], "tag": ["B-BT", "I-XX"]}
    )
    valid, issues = check_data_integrity(df)
    assert not valid


# --- filter_checked ---

def test_filter_checked_removes_unchecked():
    df = pd.DataFrame(
        {
            "True?": [True, False, True],
            "False?": [False, False, True],
            "NER_Tags": ["[1,2,3]", "[4,5,6]", "[7,8,9]"],
            "PMCID": ["A", "B", "C"],
            "Sentence": ["s1", "s2", "s3"],
        }
    )
    result = filter_checked(df)
    assert len(result) == 2
    assert "B" not in result["PMCID"].values


def test_filter_checked_nullifies_false():
    df = pd.DataFrame(
        {
            "True?": [False],
            "False?": [True],
            "NER_Tags": ["[1,2,3]"],
            "PMCID": ["A"],
            "Sentence": ["s1"],
        }
    )
    result = filter_checked(df)
    assert result.iloc[0]["NER_Tags"] is None or pd.isna(result.iloc[0]["NER_Tags"])


# --- negatives survive pipeline ---

def test_negatives_survive_parse_ner_tags():
    df = pd.DataFrame(
        {
            "True?": [True, False],
            "False?": [False, True],
            "NER_Tags": ["(4, 9, 'BLAST', 'subtiwiki')", "(48, 52, 'Fiji', 'fiji')"],
            "PMCID": ["PMC1", "PMC2"],
            "Sentence": ["Use BLAST for search", "We used Fiji for imaging"],
        }
    )
    filtered = filter_checked(df)
    parsed = parse_ner_tags(filtered)
    assert len(parsed) == 2
    negative_row = parsed[parsed["PMCID"] == "PMC2"].iloc[0]
    assert negative_row["NER_Tags"] is None


def test_negatives_get_all_O_tags(tokenizer):
    text = "We used Fiji for imaging"
    result = convert_to_iob([text], [None], tokenizer)
    tags = [tag for _, tag in result[0]]
    assert all(t == "O" for t in tags)


# --- split_by_pmcid ---

def _make_split_df(n: int) -> pd.DataFrame:
    pmcids = [f"PMC{i:04d}" for i in range(n)]
    return pd.DataFrame(
        {
            "PMCID": pmcids,
            "Sentence": [f"sentence {i}" for i in range(n)],
            "NER_Tags": [None] * n,
        }
    )


def test_split_no_pmcid_leakage():
    df = _make_split_df(100)
    train, val, test = split_by_pmcid(df, val_test_size=0.4, test_ratio_of_remainder=0.5)
    assert "PMCID" not in train.columns
    assert "PMCID" not in val.columns
    assert "PMCID" not in test.columns


def test_split_ratios():
    df = _make_split_df(100)
    train, val, test = split_by_pmcid(df, val_test_size=0.4, test_ratio_of_remainder=0.5)
    total = len(train) + len(val) + len(test)
    assert total <= 100
    assert len(train) > len(val)


def test_split_deterministic():
    df = _make_split_df(100)
    a_train, a_val, a_test = split_by_pmcid(df)
    b_train, b_val, b_test = split_by_pmcid(df)
    assert list(a_train["Sentence"]) == list(b_train["Sentence"])


# --- convert_IOB_transformer ---

def test_convert_iob_transformer_basic():
    flat = ["A", "B", "", "C", "D", ""]
    result = convert_IOB_transformer(flat, pattern="")
    assert result == [["A", "B"], ["C", "D"]]


def test_convert_iob_transformer_empty():
    assert convert_IOB_transformer([], pattern="") == []


# --- get_label_list ---

def test_get_label_list():
    df = pd.DataFrame({"tokens": ["a"], "ner_tags": ["B-BT"]})
    labels = get_label_list(df)
    assert "B-BT" in labels
    assert "" not in labels