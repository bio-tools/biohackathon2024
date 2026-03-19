#!/usr/bin/env python3
import re
import argparse
import sys
from pathlib import Path

import pandas as pd
import readchar


YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CLEAR_SCREEN = "\033[2J\033[H"


def parse_all_tags(ner_tags_str: str) -> list[tuple]:
    results = []
    for m in re.finditer(r"\((\d+),\s*(\d+),\s*'([^']*)',\s*'([^']*)'\)", str(ner_tags_str)):
        results.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
    if not results:
        for m in re.finditer(r"\((\d+),\s*(\d+),\s*'([^']*)'\)", str(ner_tags_str)):
            results.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(3).lower()))
    return results


def format_tag(tag: tuple) -> str:
    return f"({tag[0]}, {tag[1]}, '{tag[2]}', '{tag[3]}')"


def format_tags(tags: list[tuple]) -> str:
    return "; ".join(format_tag(t) for t in tags)


def highlight_entity(sentence: str, start: int, end: int) -> str:
    return sentence[:start] + YELLOW + BOLD + sentence[start:end] + RESET + sentence[end:]


def split_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        is_unannotated = row["True?"] == False and row["False?"] == False  # noqa: E712
        tags = parse_all_tags(str(row.get("NER_Tags", "")))
        if is_unannotated and len(tags) > 1:
            for tag in tags:
                r = row.to_dict()
                r["NER_Tags"] = format_tag(tag)
                r["True?"] = False
                r["False?"] = False
                rows.append(r)
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)


def consolidate_rows(work_df: pd.DataFrame) -> pd.DataFrame:
    groups: dict[tuple, list] = {}
    order: list[tuple] = []
    for _, row in work_df.iterrows():
        key = (row["PMCID"], row["Sentence"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    result = []
    for key in order:
        rows = groups[key]
        true_rows = [r for r in rows if r["True?"] == True]  # noqa: E712
        any_annotated = any(r["True?"] == True or r["False?"] == True for r in rows)  # noqa: E712

        base = rows[0]
        all_true_tags = []
        for r in true_rows:
            all_true_tags.extend(parse_all_tags(str(r["NER_Tags"])))
        all_true_tags.sort(key=lambda t: t[0])

        r_dict = {c: base[c] for c in work_df.columns}
        r_dict["True?"] = len(true_rows) > 0
        r_dict["False?"] = any_annotated and len(true_rows) == 0
        r_dict["NER_Tags"] = (
            format_tags(all_true_tags) if all_true_tags
            else (str(base["NER_Tags"]) if not any_annotated else "")
        )
        result.append(r_dict)

    return pd.DataFrame(result, columns=work_df.columns)


def split_rows_from_column(df: pd.DataFrame, spans_col: str) -> pd.DataFrame:
    """Create one work row per span in spans_col; carry original row index as _orig_idx."""
    rows = []
    for orig_idx, row in df.iterrows():
        tags = parse_all_tags(str(row.get(spans_col, "")))
        for tag in tags:
            r = row.to_dict()
            r["_orig_idx"] = orig_idx
            r["_span_tag"] = format_tag(tag)
            r["_span_accepted"] = False
            r["_span_rejected"] = False
            rows.append(r)
    cols = list(df.columns) + ["_orig_idx", "_span_tag", "_span_accepted", "_span_rejected"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def consolidate_spans(work_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Merge accepted spans back into NER_Tags of original_df rows."""
    result = original_df.copy()
    for orig_idx, group in work_df.groupby("_orig_idx"):
        accepted = group[group["_span_accepted"] == True]  # noqa: E712
        new_tags = []
        for _, r in accepted.iterrows():
            new_tags.extend(parse_all_tags(str(r["_span_tag"])))
        if new_tags:
            existing = parse_all_tags(str(original_df.at[orig_idx, "NER_Tags"]))
            merged = sorted({*existing, *new_tags}, key=lambda t: t[0])
            result.at[orig_idx, "NER_Tags"] = format_tags(merged)
            result.at[orig_idx, "True?"] = True
    return result


def run_annotation(csv_path: Path) -> None:
    work_path = csv_path.with_stem(csv_path.stem + "_work")

    if work_path.exists():
        df = pd.read_csv(work_path)
        print(f"Resuming from {work_path}")
    else:
        df = pd.read_csv(csv_path)
        required = {"PMCID", "Sentence", "True?", "False?", "NER_Tags"}
        if not required.issubset(df.columns):
            print(f"Missing columns: {required - set(df.columns)}")
            sys.exit(1)
        df = split_rows(df)
        df.to_csv(work_path, index=False)

    pending = df[(df["True?"] == False) & (df["False?"] == False)].index.tolist()  # noqa: E712
    if not pending:
        print("No unannotated rows found.")
        _finish(df, work_path, csv_path)
        return

    _annotate_loop(df, pending, span_col=None)
    df.to_csv(work_path, index=False)
    _finish(df, work_path, csv_path)


def run_spans_annotation(csv_path: Path, spans_col: str) -> None:
    work_path = csv_path.with_stem(csv_path.stem + f"_{spans_col}_work")

    if work_path.exists():
        work_df = pd.read_csv(work_path)
        print(f"Resuming from {work_path}")
    else:
        original_df = pd.read_csv(csv_path)
        if spans_col not in original_df.columns:
            print(f"Column '{spans_col}' not found. Available: {original_df.columns.tolist()}")
            sys.exit(1)
        work_df = split_rows_from_column(original_df, spans_col)
        work_df.to_csv(work_path, index=False)

    pending = work_df[
        (work_df["_span_accepted"] == False) & (work_df["_span_rejected"] == False)  # noqa: E712
    ].index.tolist()

    if not pending:
        print("No unannotated spans found.")
        _finish_spans(work_df, work_path, csv_path)
        return

    print(CLEAR_SCREEN, end="")
    print(f"{len(pending)} spans to review.\n")
    print(f"  {GREEN}\u2191 Up{RESET} = Accept    {RED}\u2193 Down{RESET} = Reject    {DIM}Backspace{RESET} = Back    {DIM}Q{RESET} = Quit\n")

    pos = 0
    while 0 <= pos < len(pending):
        idx = pending[pos]
        row = work_df.loc[idx]
        tags = parse_all_tags(str(row["_span_tag"]))

        print(f"[{pos + 1}/{len(pending)}] {DIM}{row['PMCID']}{RESET}")
        if tags:
            start, end, surface, norm = tags[0]
            highlighted = highlight_entity(str(row["Sentence"]), start, end)
            label = f"{GREEN}{surface}{RESET}"
            if norm != surface.lower():
                label += f" ({DIM}{norm}{RESET})"
            print(f"  Entity: {label}")
            print(f"  {highlighted}")
        else:
            print(f"  {row['Sentence']}")

        topics = row["Topics"] if "Topics" in row.index else None
        if topics and not isinstance(topics, float):
            print(f"  {ITALIC}{DIM}Topics: {topics}{RESET}")

        try:
            key = readchar.readkey()
        except KeyboardInterrupt:
            print("\nSaving and exiting...")
            break

        if key == readchar.key.UP:
            work_df.at[idx, "_span_accepted"] = True
            work_df.at[idx, "_span_rejected"] = False
            print(f"  \u2192 {GREEN}Accepted{RESET}\n")
            pos += 1
        elif key == readchar.key.DOWN:
            work_df.at[idx, "_span_accepted"] = False
            work_df.at[idx, "_span_rejected"] = True
            print(f"  \u2192 {RED}Rejected{RESET}\n")
            pos += 1
        elif key in (readchar.key.BACKSPACE, '\x7f', '\x08'):
            if pos > 0:
                pos -= 1
                prev_idx = pending[pos]
                work_df.at[prev_idx, "_span_accepted"] = False
                work_df.at[prev_idx, "_span_rejected"] = False
                print(f"  \u2192 {DIM}Back{RESET}\n")
            else:
                print(f"  {DIM}Already at first entry.{RESET}\n")
        elif key.lower() == "q":
            print()
            break
        else:
            print(f"  {DIM}Use \u2191/\u2193/Backspace/Q{RESET}\n")

    work_df.to_csv(work_path, index=False)
    _finish_spans(work_df, work_path, csv_path)


def _annotate_loop(df: pd.DataFrame, pending: list[int], span_col: None) -> None:
    print(CLEAR_SCREEN, end="")
    print(f"{len(pending)} rows to annotate.\n")
    print(f"  {GREEN}\u2191 Up{RESET} = Yes    {RED}\u2193 Down{RESET} = No    {DIM}Backspace{RESET} = Back    {DIM}Q{RESET} = Quit\n")

    pos = 0
    while 0 <= pos < len(pending):
        idx = pending[pos]
        row = df.loc[idx]
        tags = parse_all_tags(str(row["NER_Tags"]))

        print(f"[{pos + 1}/{len(pending)}] {DIM}{row['PMCID']}{RESET}")
        if tags:
            start, end, surface, norm = tags[0]
            highlighted = highlight_entity(str(row["Sentence"]), start, end)
            label = f"{GREEN}{surface}{RESET}"
            if norm != surface.lower():
                label += f" ({DIM}{norm}{RESET})"
            print(f"  Entity: {label}")
            print(f"  {highlighted}")
        else:
            print(f"  {row['Sentence']}")
            print(f"  NER_Tags: {row['NER_Tags']}")

        topics = row["Topics"] if "Topics" in row.index else None
        if topics and not isinstance(topics, float):
            print(f"  {ITALIC}{DIM}Topics: {topics}{RESET}")

        try:
            key = readchar.readkey()
        except KeyboardInterrupt:
            print("\nSaving and exiting...")
            break

        if key == readchar.key.UP:
            df.at[idx, "True?"] = True
            df.at[idx, "False?"] = False
            print(f"  \u2192 {GREEN}Yes{RESET}\n")
            pos += 1
        elif key == readchar.key.DOWN:
            df.at[idx, "True?"] = False
            df.at[idx, "False?"] = True
            print(f"  \u2192 {RED}No{RESET}\n")
            pos += 1
        elif key in (readchar.key.BACKSPACE, '\x7f', '\x08'):
            if pos > 0:
                pos -= 1
                prev_idx = pending[pos]
                df.at[prev_idx, "True?"] = False
                df.at[prev_idx, "False?"] = False
                print(f"  \u2192 {DIM}Back{RESET}\n")
            else:
                print(f"  {DIM}Already at first entry.{RESET}\n")
        elif key.lower() == "q":
            print()
            break
        else:
            print(f"  {DIM}Use \u2191/\u2193/Backspace/Q{RESET}\n")


def _finish(work_df: pd.DataFrame, work_path: Path, csv_path: Path) -> None:
    remaining = len(work_df[(work_df["True?"] == False) & (work_df["False?"] == False)])  # noqa: E712
    consolidated = consolidate_rows(work_df)
    consolidated.to_csv(csv_path, index=False)
    print(f"Consolidated \u2192 {csv_path}  (remaining unannotated: {remaining})")
    if remaining == 0:
        work_path.unlink(missing_ok=True)
    else:
        print(f"Progress saved to {work_path}")


def _finish_spans(work_df: pd.DataFrame, work_path: Path, csv_path: Path) -> None:
    remaining = len(
        work_df[(work_df["_span_accepted"] == False) & (work_df["_span_rejected"] == False)]  # noqa: E712
    )
    original_df = pd.read_csv(csv_path)
    updated = consolidate_spans(work_df, original_df)
    updated.to_csv(csv_path, index=False)
    accepted = int((work_df["_span_accepted"] == True).sum())  # noqa: E712
    print(f"Merged {accepted} accepted spans \u2192 {csv_path}  (remaining unreviewed: {remaining})")
    if remaining == 0:
        work_path.unlink(missing_ok=True)
    else:
        print(f"Progress saved to {work_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate unannotated NER rows")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--spans-column",
        metavar="COL",
        help="Annotate spans from this column (e.g. Review_Spans, Added_Spans) instead of NER_Tags",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"File not found: {args.csv_path}")
        sys.exit(1)

    if args.spans_column:
        run_spans_annotation(args.csv_path, args.spans_column)
    else:
        run_annotation(args.csv_path)


if __name__ == "__main__":
    main()
