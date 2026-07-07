import ast
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
LATEX_RE = re.compile(r"\\documentclass")
MAX_SENTENCE_LEN = 512
MIN_SENTENCE_WORDS = 5


def strip_non_ascii(text: str) -> tuple[str, dict[int, int]]:
    new_chars: list[str] = []
    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old_idx, ch in enumerate(text):
        old_to_new[old_idx] = new_idx
        if ch.isascii():
            new_chars.append(ch)
            new_idx += 1
    old_to_new[len(text)] = new_idx
    return "".join(new_chars), old_to_new


def parse_tags(raw: str) -> list[tuple]:
    results = []
    for span_str in raw.split("; "):
        span_str = span_str.strip()
        if not span_str:
            continue
        try:
            parsed = ast.literal_eval(span_str)
            if isinstance(parsed, tuple) and len(parsed) >= 4:
                results.append(parsed)
        except (ValueError, SyntaxError):
            pass
    return results


def tag_overlaps_url(sentence: str, start: int, end: int) -> bool:
    for m in URL_RE.finditer(sentence):
        if start < m.end() and end > m.start():
            return True
    return False


def has_letter_boundary(sentence: str, start: int, end: int) -> bool:
    if start > 0 and sentence[start - 1].isalpha():
        return True
    if end < len(sentence) and sentence[end].isalpha():
        return True
    return False


def has_latex(sentence: str) -> bool:
    return bool(LATEX_RE.search(sentence))


def try_realign(sentence: str, name: str, start: int) -> tuple[int, int] | None:
    pattern = re.compile(r"(?<![a-zA-Z])" + re.escape(name) + r"(?![a-zA-Z])", re.IGNORECASE)
    matches = list(pattern.finditer(sentence))
    if len(matches) == 1:
        return matches[0].start(), matches[0].end()
    if matches:
        closest = min(matches, key=lambda m: abs(m.start() - start))
        return closest.start(), closest.end()
    return None


def filter_short_sentences(df: pd.DataFrame, min_words: int = MIN_SENTENCE_WORDS) -> pd.DataFrame:
    mask = df["Sentence"].str.split().str.len().ge(min_words)
    dropped = (~mask).sum()
    if dropped:
        logger.info("Dropped %d sentences with fewer than %d words", dropped, min_words)
    return df[mask].reset_index(drop=True)


def clean(
    csv_path: Path,
    max_len: int = MAX_SENTENCE_LEN,
    min_words: int = MIN_SENTENCE_WORDS,
    dry_run: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    n_orig = len(df)

    drop_idx: set = set()
    realigned = 0
    non_ascii_stripped = 0
    reasons: dict[str, int] = {}

    def mark_drop(idx: object, reason: str) -> None:
        drop_idx.add(idx)
        reasons[reason] = reasons.get(reason, 0) + 1

    def _format_tags(tags: list[tuple]) -> str:
        return "; ".join(str(t) for t in tags)

    def _remap_tags(tags: list[tuple], pos_map: dict[int, int], sentence: str) -> list[tuple] | None:
        remapped = []
        for t in tags:
            ns, ne = pos_map[int(t[0])], pos_map[int(t[1])]
            if ns >= ne:
                return None
            remapped.append((ns, ne, sentence[ns:ne], t[3]))
        return remapped

    for idx, row in df.iterrows():
        sentence = str(row["Sentence"])
        raw_tags = row.get("NER_Tags")

        cleaned, pos_map = strip_non_ascii(sentence)
        if cleaned != sentence:
            non_ascii_stripped += 1
            sentence = cleaned
            df.at[idx, "Sentence"] = sentence
            if not pd.isna(raw_tags):
                tags = parse_tags(str(raw_tags))
                if tags:
                    remapped = _remap_tags(tags, pos_map, sentence)
                    if remapped is None:
                        mark_drop(idx, "span_in_non_ascii")
                        continue
                    df.at[idx, "NER_Tags"] = _format_tags(remapped)
                    raw_tags = df.at[idx, "NER_Tags"]
            if "NER_tags_predicted" in df.columns:
                raw_pred = row.get("NER_tags_predicted")
                if isinstance(raw_pred, str) and raw_pred.strip():
                    pred_tags = parse_tags(raw_pred)
                    if pred_tags:
                        remapped = _remap_tags(pred_tags, pos_map, sentence)
                        df.at[idx, "NER_tags_predicted"] = _format_tags(remapped) if remapped else ""

        if len(sentence.split()) < min_words:
            mark_drop(idx, "sentence_too_short")
            continue

        if len(sentence) > max_len:
            mark_drop(idx, "sentence_too_long")
            continue

        if has_latex(sentence):
            mark_drop(idx, "latex_content")
            continue

        if pd.isna(raw_tags):
            continue

        tags = parse_tags(str(raw_tags))
        if not tags:
            mark_drop(idx, "unparseable_tag")
            continue

        drop_row = False
        kept_tags = []
        for tag in tags:
            start, end, name = int(tag[0]), int(tag[1]), str(tag[2])

            if start < 0 or end < 0 or start >= end or end > len(sentence):
                mark_drop(idx, "invalid_span")
                drop_row = True
                break

            if tag_overlaps_url(sentence, start, end):
                mark_drop(idx, "tag_in_url")
                drop_row = True
                break

            if has_letter_boundary(sentence, start, end):
                new_span = try_realign(sentence, name, start)
                if new_span is not None:
                    new_start, new_end = new_span
                    if not has_letter_boundary(sentence, new_start, new_end):
                        extracted = sentence[new_start:new_end]
                        kept_tags.append((new_start, new_end, extracted, tag[3]))
                        realigned += 1
                        continue
                mark_drop(idx, "tag_mid_word")
                drop_row = True
                break

            kept_tags.append(tag)

        if drop_row:
            continue
        if kept_tags:
            df.at[idx, "NER_Tags"] = _format_tags(kept_tags)

    df_clean = df.drop(index=list(drop_idx)).reset_index(drop=True)

    prefix = "[DRY RUN] " if dry_run else ""
    logger.info("%sOriginal rows: %d", prefix, n_orig)
    logger.info("%sNon-ASCII sentences stripped: %d", prefix, non_ascii_stripped)
    logger.info("%sDropped rows: %d", prefix, len(drop_idx))
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        logger.info("%s  %s: %d", prefix, reason, count)
    logger.info("%sRealigned tags: %d", prefix, realigned)
    logger.info("%sRemaining rows: %d", prefix, len(df_clean))

    if dry_run:
        return df

    return df_clean
