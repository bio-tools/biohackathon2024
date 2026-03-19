#!/usr/bin/env python3
"""Review predicted NER spans against original NER_Tags; confirm or reject each span."""

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
CYAN = "\033[96m"
CLEAR_SCREEN = "\033[2J\033[H"


def parse_all_tags(tag_str: str) -> list[tuple]:
    results = []
    for m in re.finditer(r"\((\d+),\s*(\d+),\s*'([^']*)',\s*'([^']*)'\)", str(tag_str)):
        results.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
    if not results:
        for m in re.finditer(r"\((\d+),\s*(\d+),\s*'([^']*)'\)", str(tag_str)):
            results.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(3).lower()))
    return results


def format_tag(tag: tuple) -> str:
    return f"({tag[0]}, {tag[1]}, '{tag[2]}', '{tag[3]}')"


def format_tags(tags: list[tuple]) -> str:
    return "; ".join(format_tag(t) for t in tags)


def highlight_spans(sentence: str, spans: list[tuple], color: str) -> str:
    result = []
    prev = 0
    for start, end, *_ in sorted(spans, key=lambda t: t[0]):
        result.append(sentence[prev:start])
        result.append(color + BOLD + sentence[start:end] + RESET)
        prev = end
    result.append(sentence[prev:])
    return "".join(result)


def build_work_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for orig_idx, row in df.iterrows():
        pred_tags = parse_all_tags(str(row.get("NER_tags_predicted", "")))
        if not pred_tags:
            continue
        for tag in pred_tags:
            r = row.to_dict()
            r["_orig_idx"] = orig_idx
            r["_pred_span"] = format_tag(tag)
            r["_accepted"] = False
            r["_rejected"] = False
            rows.append(r)
    cols = list(df.columns) + ["_orig_idx", "_pred_span", "_accepted", "_rejected"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def consolidate(work_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    result = original_df.copy()
    for orig_idx, group in work_df.groupby("_orig_idx"):
        accepted = group[group["_accepted"] == True]  # noqa: E712
        new_tags = []
        for _, r in accepted.iterrows():
            new_tags.extend(parse_all_tags(str(r["_pred_span"])))
        if new_tags:
            new_tags_sorted = sorted(new_tags, key=lambda t: t[0])
            result.at[orig_idx, "NER_Tags"] = format_tags(new_tags_sorted)
    return result


def run(csv_path: Path) -> None:
    work_path = csv_path.with_stem(csv_path.stem + "_postpredict_work")
    out_path = csv_path.with_stem(csv_path.stem + "_annotated")

    if work_path.exists():
        work_df = pd.read_csv(work_path)
        print(f"Resuming from {work_path}")
    else:
        df = pd.read_csv(csv_path)
        required = {"PMCID", "Sentence", "NER_Tags", "NER_tags_predicted"}
        if not required.issubset(df.columns):
            print(f"Missing columns: {required - set(df.columns)}")
            sys.exit(1)
        work_df = build_work_df(df)
        work_df.to_csv(work_path, index=False)

    pending = work_df[
        (work_df["_accepted"] == False) & (work_df["_rejected"] == False)  # noqa: E712
    ].index.tolist()

    if not pending:
        print("Nothing to review.")
        _finish(work_df, work_path, csv_path, out_path)
        return

    print(CLEAR_SCREEN, end="")
    print(f"{len(pending)} predicted spans to review.\n")
    print(
        f"  {GREEN}↑ Up{RESET} = Accept    "
        f"{RED}↓ Down{RESET} = Reject    "
        f"{DIM}Backspace{RESET} = Back    "
        f"{DIM}Q{RESET} = Quit\n"
    )

    pos = 0
    while 0 <= pos < len(pending):
        idx = pending[pos]
        row = work_df.loc[idx]
        sentence = str(row["Sentence"])

        pred_tags = parse_all_tags(str(row["_pred_span"]))
        orig_tags = parse_all_tags(str(row.get("NER_Tags", "")))

        print(f"[{pos + 1}/{len(pending)}] {DIM}{row['PMCID']}{RESET}")

        if pred_tags:
            start, end, surface, norm = pred_tags[0]
            pred_highlighted = highlight_spans(sentence, pred_tags, YELLOW)
            label = f"{YELLOW}{BOLD}{surface}{RESET}"
            if norm != surface.lower():
                label += f" ({DIM}{norm}{RESET})"
            print(f"  Predicted: {label}")
            print(f"  {pred_highlighted}")
        else:
            print(f"  {sentence}")

        if orig_tags:
            orig_highlighted = highlight_spans(sentence, orig_tags, CYAN)
            orig_label = format_tags(orig_tags)
            print(f"  {DIM}Original:  {orig_label}{RESET}")
            print(f"  {orig_highlighted}")
        else:
            print(f"  {DIM}Original:  (none){RESET}")

        topics = row.get("Topics")
        if topics and not isinstance(topics, float):
            print(f"  {ITALIC}{DIM}Topics: {topics}{RESET}")

        already_accepted = bool(work_df.at[idx, "_accepted"])
        already_rejected = bool(work_df.at[idx, "_rejected"])
        if already_accepted:
            print(f"  {DIM}[Currently: {GREEN}accepted{RESET}{DIM} — ↑ keep / ↓ change]{RESET}")
        elif already_rejected:
            print(f"  {DIM}[Currently: {RED}rejected{RESET}{DIM} — ↓ keep / ↑ change]{RESET}")

        try:
            key = readchar.readkey()
        except KeyboardInterrupt:
            print("\nSaving and exiting...")
            break

        if key == readchar.key.UP:
            work_df.at[idx, "_accepted"] = True
            work_df.at[idx, "_rejected"] = False
            print(f"  → {GREEN}Accepted{RESET}\n")
            pos += 1
            work_df.to_csv(work_path, index=False)
        elif key == readchar.key.DOWN:
            work_df.at[idx, "_accepted"] = False
            work_df.at[idx, "_rejected"] = True
            print(f"  → {RED}Rejected{RESET}\n")
            pos += 1
            work_df.to_csv(work_path, index=False)
        elif key in (readchar.key.BACKSPACE, "\x7f", "\x08"):
            if pos > 0:
                pos -= 1
                print(f"  → {DIM}Back{RESET}\n")
            else:
                print(f"  {DIM}Already at first entry.{RESET}\n")
        elif key.lower() == "q":
            print()
            break
        else:
            print(f"  {DIM}Use ↑/↓/Backspace/Q{RESET}\n")

    work_df.to_csv(work_path, index=False)
    _finish(work_df, work_path, csv_path, out_path)


def _finish(work_df: pd.DataFrame, work_path: Path, csv_path: Path, out_path: Path) -> None:
    remaining = int(
        ((work_df["_accepted"] == False) & (work_df["_rejected"] == False)).sum()  # noqa: E712
    )
    if csv_path.exists():
        original_df = pd.read_csv(csv_path)
    else:
        internal = {"_orig_idx", "_pred_span", "_accepted", "_rejected"}
        orig_cols = [c for c in work_df.columns if c not in internal]
        original_df = (
            work_df[["_orig_idx"] + orig_cols]
            .drop_duplicates(subset=["_orig_idx"])
            .set_index("_orig_idx")
            .rename_axis(None)
        )
        original_df.index = original_df.index.astype(int)
    updated = consolidate(work_df, original_df)
    updated.to_csv(out_path, index=False)
    accepted = int((work_df["_accepted"] == True).sum())  # noqa: E712
    print(f"Merged {accepted} accepted spans → {out_path}  (remaining unreviewed: {remaining})")
    if remaining == 0:
        work_path.unlink(missing_ok=True)
    else:
        print(f"Progress saved to {work_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review predicted NER spans")
    parser.add_argument("csv_path", type=Path, help="Predicted CSV with NER_tags_predicted column")
    args = parser.parse_args()

    work_path = args.csv_path.with_stem(args.csv_path.stem + "_postpredict_work")
    if not args.csv_path.exists() and not work_path.exists():
        print(f"File not found: {args.csv_path}")
        sys.exit(1)

    run(args.csv_path)


if __name__ == "__main__":
    main()
