#!/usr/bin/env python3
import argparse
import ast
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


def highlight_span(sentence: str, start: int, end: int) -> str:
    return sentence[:start] + YELLOW + BOLD + sentence[start:end] + RESET + sentence[end:]


def parse_span(ner_tags_str: str) -> tuple[int, int, str] | None:
    try:
        parsed = ast.literal_eval(ner_tags_str)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, tuple) and len(parsed) >= 3:
        return int(parsed[0]), int(parsed[1]), str(parsed[2])
    return None


def run_annotation(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    required = {"PMCID", "Sentence", "True?", "False?", "NER_Tags"}
    if not required.issubset(df.columns):
        print(f"Missing columns: {required - set(df.columns)}")
        sys.exit(1)

    pending = df[(df["True?"] == False) & (df["False?"] == False)].index.tolist()  # noqa: E712
    if not pending:
        print("No unannotated rows found.")
        return

    print(CLEAR_SCREEN, end="")
    print(f"{len(pending)} rows to annotate.\n")
    print(f"  {GREEN}\u2191 Up{RESET} = Yes (positive)    {RED}\u2193 Down{RESET} = No (negative)    {DIM}Backspace{RESET} = Back    {DIM}Q{RESET} = Quit\n")

    pos = 0
    while 0 <= pos < len(pending):
        idx = pending[pos]
        row = df.loc[idx]
        span = parse_span(str(row["NER_Tags"]))

        print(f"[{pos + 1}/{len(pending)}] {DIM}{row['PMCID']}{RESET}")
        if span:
            start, end, entity = span
            print(f"  Entity: {GREEN}{entity}{RESET}")
            print(f"  {highlight_span(row['Sentence'], start, end)}")
        else:
            print(f"  {row['Sentence']}")
            print(f"  NER_Tags: {row['NER_Tags']}")

        topics = row.get("Topics") if hasattr(row, "get") else row["Topics"] if "Topics" in row.index else None
        if topics and not (isinstance(topics, float)):
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
                print(f"  \u2192 {DIM}Back{RESET}\n")
            else:
                print(f"  {DIM}Already at first entry.{RESET}\n")
        elif key.lower() == "q":
            print()
            break
        else:
            print(f"  {DIM}Use \u2191/\u2193/Backspace/Q{RESET}\n")

    df.to_csv(csv_path, index=False)
    remaining = len(df[(df["True?"] == False) & (df["False?"] == False)])  # noqa: E712
    print(f"Saved to {csv_path}. Remaining unannotated: {remaining}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate unannotated NER rows")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"File not found: {args.csv_path}")
        sys.exit(1)

    run_annotation(args.csv_path)


if __name__ == "__main__":
    main()
