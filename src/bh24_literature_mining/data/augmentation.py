import logging
import random
from pathlib import Path

import pandas as pd

from bh24_literature_mining.utils import load_biotools_pub

logger = logging.getLogger(__name__)

MIN_TOOL_NAME_LEN = 3


def build_tool_vocab(biotools_path: Path, min_len: int = MIN_TOOL_NAME_LEN) -> list[str]:
    df = load_biotools_pub(str(biotools_path))
    names = df["name"].dropna().unique().tolist()
    filtered = [
        n for n in names
        if len(n) >= min_len and n.replace("-", "").replace("_", "").isalnum()
    ]
    return sorted(set(filtered))


def _count_tokens(name: str) -> int:
    return len(name.split())


def substitute_entity(
    sentence: str,
    ner_tags: list[list],
    target_idx: int,
    new_name: str,
) -> tuple[str, list[list]]:
    tag = ner_tags[target_idx]
    old_start, old_end = tag[0], tag[1]
    old_name = tag[2]
    delta = len(new_name) - len(old_name)

    new_sentence = sentence[:old_start] + new_name + sentence[old_end:]

    new_tags = []
    for i, t in enumerate(ner_tags):
        start, end, name, etype = t[0], t[1], t[2], t[3]
        if i == target_idx:
            new_tags.append([old_start, old_start + len(new_name), new_name, etype])
        elif start >= old_end:
            new_tags.append([start + delta, end + delta, name, etype])
        else:
            new_tags.append([start, end, name, etype])

    return new_sentence, new_tags


def _validate_substitution(sentence: str, ner_tags: list[list]) -> bool:
    for tag in ner_tags:
        start, end = tag[0], tag[1]
        if start < 0 or end > len(sentence) or start >= end:
            return False
    for i in range(len(ner_tags)):
        for j in range(i + 1, len(ner_tags)):
            a_start, a_end = ner_tags[i][0], ner_tags[i][1]
            b_start, b_end = ner_tags[j][0], ner_tags[j][1]
            if a_start < b_end and b_start < a_end:
                return False
    return True


def augment_dataframe(
    df: pd.DataFrame,
    tool_vocab: list[str],
    n_copies: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    vocab_by_token_count: dict[int, list[str]] = {}
    for name in tool_vocab:
        tc = _count_tokens(name)
        vocab_by_token_count.setdefault(tc, []).append(name)

    augmented_rows: list[dict] = []

    for _, row in df.iterrows():
        ner_tags = row["NER_Tags"]
        if not ner_tags:
            continue

        sentence = row["Sentence"]
        entity_indices = list(range(len(ner_tags)))

        generated = 0
        attempts = 0
        max_attempts = n_copies * 5

        while generated < n_copies and attempts < max_attempts:
            attempts += 1
            target_idx = rng.choice(entity_indices)
            tag = ner_tags[target_idx]
            old_name = tag[2]
            tc = _count_tokens(old_name)

            candidates = vocab_by_token_count.get(tc, [])
            if not candidates:
                continue

            new_name = rng.choice(candidates)
            if new_name == old_name:
                continue
            if new_name.lower() in sentence.lower():
                continue

            new_sentence, new_tags = substitute_entity(
                sentence, ner_tags, target_idx, new_name
            )

            if not _validate_substitution(new_sentence, new_tags):
                logger.debug("Invalid substitution: %s -> %s", old_name, new_name)
                continue

            augmented_rows.append({
                "Sentence": new_sentence,
                "NER_Tags": new_tags,
            })
            generated += 1

    if not augmented_rows:
        return pd.DataFrame(columns=["Sentence", "NER_Tags"])

    aug_df = pd.DataFrame(augmented_rows)
    logger.info(
        "Generated %d augmented sentences from %d originals",
        len(aug_df),
        len(df[df["NER_Tags"].apply(bool)]),
    )
    return aug_df
