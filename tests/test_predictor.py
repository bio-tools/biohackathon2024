from bh24_literature_mining.inference.predictor import format_entities, truncate_if_needed


def test_format_entities_empty():
    assert format_entities([], "hello") == ""


def test_format_entities_single():
    entities = [{"start": 0, "end": 5, "entity_group": "BT"}]
    result = format_entities(entities, "BLAST is great")
    assert "BLAST" in result
    assert "BT" in result
    assert "0-5" in result


def test_format_entities_multiple():
    entities = [
        {"start": 0, "end": 5, "entity_group": "BT"},
        {"start": 10, "end": 18, "entity_group": "BT"},
    ]
    result = format_entities(entities, "BLAST and MaxQuant are tools")
    assert ";" in result


def test_truncate_short_sentence(tokenizer_fixture):
    result = truncate_if_needed("short sentence", tokenizer_fixture, max_length=512)
    assert result == "short sentence"


def test_truncate_long_sentence(tokenizer_fixture):
    long_text = "word " * 1000
    result = truncate_if_needed(long_text, tokenizer_fixture, max_length=32)
    tokens = tokenizer_fixture(result, return_tensors="pt", truncation=False)
    assert tokens["input_ids"].shape[1] <= 32


import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer_fixture():
    return AutoTokenizer.from_pretrained("bioformers/bioformer-16L")
