"""Generate corpus overview figure and source data tables."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pandas as pd

from bh24_literature_mining.data.annotation_parser import prepare_annotations
from bh24_literature_mining.data.preparation import count_negative_rows
from bh24_literature_mining.data.splitter import (
    get_dataframe_resource_ids,
    split_by_pmcid_and_resource,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

ANNOTATIONS = Path("data/annotated/260507_annotated.csv")
OUT_DIR = Path("paper/figures")
SEED = 42


def parse_tags(value: object) -> list[tuple[int, int, str, str]]:
    if pd.isna(value) or not str(value).strip():
        return []
    tags = []
    for part in str(value).split(";"):
        part = part.strip()
        if part:
            tags.append(ast.literal_eval(part))
    return tags


def iob_counts(path: Path) -> dict[str, int]:
    sentences = 0
    tokens = 0
    mentions = 0
    seen = False
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if seen:
                    sentences += 1
                seen = False
                continue
            seen = True
            tokens += 1
            label = line.split("\t")[-1]
            if label.startswith("B-"):
                mentions += 1
        if seen:
            sentences += 1
    return {"sentences": sentences, "tokens": tokens, "mentions": mentions}


def split_clean_annotations() -> dict[str, pd.DataFrame]:
    parsed = prepare_annotations(
        ANNOTATIONS,
        entity_type=None,
        include_negatives=True,
    )
    train, validation, test = split_by_pmcid_and_resource(
        parsed,
        random_seed=SEED,
        keep_pmcid=True,
    )
    return {"Train": train, "Validation": validation, "Test": test}


def build_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(ANNOTATIONS)
    positives = raw[raw["True?"].eq(True)].copy()

    mention_rows = []
    for _, row in positives.iterrows():
        for start, end, name, tool_id in parse_tags(row["NER_Tags"]):
            mention_rows.append(
                {
                    "PMCID": row["PMCID"],
                    "surface_form": name,
                    "tool_id": tool_id,
                    "mention_length": int(end) - int(start),
                }
            )
    mentions = pd.DataFrame(mention_rows)
    tool_frequency = (
        mentions["tool_id"]
        .value_counts()
        .rename_axis("tool_id")
        .reset_index(name="mentions")
    )
    tool_frequency["rank"] = range(1, len(tool_frequency) + 1)

    split_frames = split_clean_annotations()
    iob_paths = {
        "Train": Path("data/IOB_260713/train_IOB.tsv"),
        "Validation": Path("data/IOB_260713/val_IOB.tsv"),
        "Test": Path("data/IOB_260713/test_IOB.tsv"),
    }
    split_rows = []
    split_tools: dict[str, set[str]] = {}
    for split_name, frame in split_frames.items():
        split_tools[split_name] = get_dataframe_resource_ids(frame)
        counts = iob_counts(iob_paths[split_name])
        split_rows.append(
            {
                "split": split_name,
                "pmcids": frame["PMCID"].nunique(),
                "sentences": counts["sentences"],
                "negative_sentences": count_negative_rows(frame),
                "tokens": counts["tokens"],
                "mentions": counts["mentions"],
                "unique_tool_ids": len(split_tools[split_name]),
            }
        )
    split_summary = pd.DataFrame(split_rows)

    unseen_rows = []
    train_tools = split_tools["Train"]
    val_tools = split_tools["Validation"]
    test_tools = split_tools["Test"]
    for split_name, tools, baseline in [
        ("Validation", val_tools, train_tools),
        ("Test", test_tools, train_tools),
    ]:
        unseen = tools - baseline
        unseen_rows.append(
            {
                "split": split_name,
                "unseen_tool_ids": len(unseen),
                "seen_tool_ids": len(tools) - len(unseen),
                "total_tool_ids": len(tools),
                "unseen_fraction": len(unseen) / len(tools),
            }
        )
    unseen_summary = pd.DataFrame(unseen_rows)

    stats = {
        "raw_rows": int(len(raw)),
        "positive_rows": int(len(positives)),
        "negative_rows": int(raw["False?"].eq(True).sum()),
        "mentions": int(len(mentions)),
        "unique_surface_forms": int(mentions["surface_form"].nunique()),
        "unique_tool_ids": int(mentions["tool_id"].nunique()),
        "singleton_tool_ids": int((tool_frequency["mentions"] == 1).sum()),
        "singleton_tool_id_fraction": float((tool_frequency["mentions"] == 1).mean()),
        "top_50_mention_fraction": float(tool_frequency.head(50)["mentions"].sum() / len(mentions)),
        "median_mention_length": float(mentions["mention_length"].median()),
        "mean_mentions_per_positive_sentence": float(len(mentions) / len(positives)),
    }
    return split_summary, tool_frequency, unseen_summary, stats


def save_source_data(
    split_summary: pd.DataFrame,
    tool_frequency: pd.DataFrame,
    unseen_summary: pd.DataFrame,
    stats: dict[str, object],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_summary.to_csv(OUT_DIR / "split_summary.csv", index=False)
    tool_frequency.to_csv(OUT_DIR / "tool_frequency.csv", index=False)
    unseen_summary.to_csv(OUT_DIR / "unseen_tool_summary.csv", index=False)
    (OUT_DIR / "dataset_overview_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n"
    )


def plot_figure(
    split_summary: pd.DataFrame,
    tool_frequency: pd.DataFrame,
    unseen_summary: pd.DataFrame,
    stats: dict[str, object],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    metrics = ["pmcids", "sentences", "mentions", "unique_tool_ids"]
    labels = ["PMCIDs", "Sentences", "Mentions", "Resource IDs"]
    x = range(len(split_summary))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for metric, label, color, offset in zip(metrics, labels, colors, offsets):
        values = split_summary[metric].to_numpy()
        ax_a.bar([i + offset for i in x], values, width=width, label=label, color=color)
    ax_a.set_xticks(list(x))
    ax_a.set_xticklabels(split_summary["split"])
    ax_a.set_ylabel("Count")
    ax_a.set_ylim(0, split_summary[metrics].to_numpy().max() * 1.15)
    ax_a.set_title(
        "A. Document- and resource-disjoint splits",
        loc="left",
        fontweight="bold",
    )
    ax_a.legend(frameon=False, ncols=2)
    ax_a.spines[["top", "right"]].set_visible(False)

    ax_b.plot(
        tool_frequency["rank"],
        tool_frequency["mentions"],
        color="#4C78A8",
        linewidth=1.6,
    )
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Normalized resource ID rank")
    ax_b.set_ylabel("Mentions")
    ax_b.set_title(
        "B. Long-tail resource ID distribution", loc="left", fontweight="bold"
    )
    ax_b.text(
        0.98,
        0.95,
        f"{100 * stats['singleton_tool_id_fraction']:.1f}% singleton IDs\n"
        f"Top 50 IDs: {100 * stats['top_50_mention_fraction']:.1f}% of mentions",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "0.8", "pad": 3},
    )
    ax_b.spines[["top", "right"]].set_visible(False)

    raw = pd.read_csv(ANNOTATIONS)
    lengths = []
    mentions_per_sentence = []
    for _, row in raw[raw["True?"].eq(True)].iterrows():
        tags = parse_tags(row["NER_Tags"])
        if tags:
            mentions_per_sentence.append(len(tags))
            lengths.extend(int(end) - int(start) for start, end, _, _ in tags)
    ax_c.hist(lengths, bins=range(1, 34), color="#72B7B2", edgecolor="white")
    ax_c.axvline(stats["median_mention_length"], color="black", linestyle="--", linewidth=1)
    ax_c.set_xlabel("Mention length (characters)")
    ax_c.set_ylabel("Mentions")
    ax_c.set_title("C. Mention length distribution", loc="left", fontweight="bold")
    ax_c.text(
        0.98,
        0.95,
        f"Median = {stats['median_mention_length']:.0f} characters",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "0.8", "pad": 3},
    )
    ax_c.spines[["top", "right"]].set_visible(False)

    labels_d = unseen_summary["split"].tolist()
    seen = unseen_summary["seen_tool_ids"].to_numpy()
    unseen = unseen_summary["unseen_tool_ids"].to_numpy()
    y = range(len(labels_d))
    ax_d.barh(y, seen, color="#B8B8B8", label="Seen")
    ax_d.barh(y, unseen, left=seen, color="#F58518", label="Unseen")
    for idx, row in unseen_summary.iterrows():
        ax_d.text(
            row["total_tool_ids"] + 25,
            idx,
            f"{100 * row['unseen_fraction']:.1f}% unseen",
            va="center",
            fontsize=7,
        )
    ax_d.set_yticks(list(y))
    ax_d.set_yticklabels(labels_d)
    ax_d.set_xlabel("Unique normalized resource IDs")
    ax_d.set_title("D. Unseen-resource composition", loc="left", fontweight="bold")
    ax_d.set_xlim(0, unseen_summary["total_tool_ids"].max() * 1.35)
    ax_d.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.7, h_pad=2.4, w_pad=2.4)
    fig.subplots_adjust(bottom=0.16)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"dataset_overview.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    split_summary, tool_frequency, unseen_summary, stats = build_data()
    save_source_data(split_summary, tool_frequency, unseen_summary, stats)
    plot_figure(split_summary, tool_frequency, unseen_summary, stats)


if __name__ == "__main__":
    main()
