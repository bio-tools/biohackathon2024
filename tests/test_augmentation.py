import pandas as pd
import pytest

from bh24_literature_mining.data.augmentation import (
    _validate_substitution,
    augment_dataframe,
    build_tool_vocab,
    substitute_entity,
)


def test_substitute_entity_basic():
    sentence = "We used BLAST for sequence alignment"
    ner_tags = [[8, 13, "BLAST", "BT"]]
    new_sentence, new_tags = substitute_entity(sentence, ner_tags, 0, "Bowtie2")
    assert new_sentence == "We used Bowtie2 for sequence alignment"
    assert new_tags[0][0] == 8
    assert new_tags[0][1] == 15
    assert new_tags[0][2] == "Bowtie2"


def test_substitute_entity_shifts_later_tags():
    sentence = "Use BLAST and MaxQuant for analysis"
    ner_tags = [
        [4, 9, "BLAST", "BT"],
        [14, 22, "MaxQuant", "BT"],
    ]
    new_sentence, new_tags = substitute_entity(sentence, ner_tags, 0, "Bowtie2")
    assert new_sentence == "Use Bowtie2 and MaxQuant for analysis"
    assert new_tags[0] == [4, 11, "Bowtie2", "BT"]
    delta = len("Bowtie2") - len("BLAST")
    assert new_tags[1] == [14 + delta, 22 + delta, "MaxQuant", "BT"]


def test_substitute_entity_does_not_shift_earlier_tags():
    sentence = "Use BLAST and MaxQuant for analysis"
    ner_tags = [
        [4, 9, "BLAST", "BT"],
        [14, 22, "MaxQuant", "BT"],
    ]
    new_sentence, new_tags = substitute_entity(sentence, ner_tags, 1, "Seurat")
    assert "BLAST" in new_sentence
    assert "Seurat" in new_sentence
    assert new_tags[0] == [4, 9, "BLAST", "BT"]
    assert new_tags[1][2] == "Seurat"


def test_substitute_shorter_name():
    sentence = "Run MaxQuant now"
    ner_tags = [[4, 12, "MaxQuant", "BT"]]
    new_sentence, new_tags = substitute_entity(sentence, ner_tags, 0, "BWA")
    assert new_sentence == "Run BWA now"
    assert new_tags[0] == [4, 7, "BWA", "BT"]


def test_validate_substitution_valid():
    assert _validate_substitution("Use BLAST here", [[4, 9, "BLAST", "BT"]])


def test_validate_substitution_out_of_bounds():
    assert not _validate_substitution("short", [[0, 100, "x", "BT"]])


def test_validate_substitution_overlapping():
    tags = [[0, 5, "A", "BT"], [3, 8, "B", "BT"]]
    assert not _validate_substitution("some sentence here", tags)


def test_augment_dataframe_produces_copies():
    df = pd.DataFrame({
        "Sentence": ["We used BLAST for search", "No entities here"],
        "NER_Tags": [[[8, 13, "BLAST", "BT"]], None],
    })
    vocab = ["Seurat", "BWA", "Bowtie2", "HISAT2", "Salmon", "GATK"]
    aug = augment_dataframe(df, vocab, n_copies=3, seed=42)
    assert len(aug) >= 1
    assert len(aug) <= 3
    for _, row in aug.iterrows():
        assert "BLAST" not in row["Sentence"] or row["NER_Tags"][0][2] != "BLAST"


def test_augment_dataframe_deterministic():
    df = pd.DataFrame({
        "Sentence": ["We used BLAST for search"],
        "NER_Tags": [[[8, 13, "BLAST", "BT"]]],
    })
    vocab = ["Seurat", "BWA", "Bowtie2", "HISAT2", "Salmon"]
    a = augment_dataframe(df, vocab, n_copies=3, seed=42)
    b = augment_dataframe(df, vocab, n_copies=3, seed=42)
    assert list(a["Sentence"]) == list(b["Sentence"])


def test_augment_skips_if_name_already_in_sentence():
    df = pd.DataFrame({
        "Sentence": ["We used BLAST and Seurat for search"],
        "NER_Tags": [[[8, 13, "BLAST", "BT"]]],
    })
    vocab = ["Seurat"]
    aug = augment_dataframe(df, vocab, n_copies=3, seed=42)
    assert len(aug) == 0


def test_augment_empty_df():
    df = pd.DataFrame({"Sentence": [], "NER_Tags": []})
    vocab = ["Tool1", "Tool2"]
    aug = augment_dataframe(df, vocab, n_copies=3, seed=42)
    assert len(aug) == 0


def test_build_tool_vocab(tmp_path):
    tsv = tmp_path / "tools.tsv"
    tsv.write_text("name\tbiotoolsID\npubmedid\npubmedcid\nlink\nEDAM_topics\n")
    lines = [
        "Seurat\tseurat\t123\tPMC1\tlink\tRNA",
        "AB\tab\t124\tPMC2\tlink\tRNA",
        "MaxQuant\tmaxquant\t125\tPMC3\tlink\tProteomics",
        "bad name!\tbad\t126\tPMC4\tlink\t",
    ]
    tsv.write_text("name\tbiotoolsID\tpubmedid\tpubmedcid\tlink\tEDAM_topics\n" + "\n".join(lines))
    vocab = build_tool_vocab(tsv, min_len=3)
    assert "Seurat" in vocab
    assert "MaxQuant" in vocab
    assert "AB" not in vocab
