"""Heatmap of HPL Gflops vs N and NB from hpl-score.csv.

Usage: python score_heatmap.py ../hpl-score.csv [-o hpl_score_heatmap.png] [--min-gpus 220]

The sheet has two row layouts (9 and 13 fields); both are handled. Only
A100_80 rows with at least --min-gpus GPUs are plotted, runs marked "cut off"
are skipped, and for duplicated (N, NB) the highest Gflops is kept.
"""
import argparse
import csv
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def first_num(s):
    m = NUM.search(s or "")
    return float(m.group(0).replace(",", "")) if m else None


def load(csv_file, min_gpus):
    recs = []
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        if len(r) >= 13:          # later layout: % R peak column inserted after Gflops
            n_s, nb_s = r[4], r[5]
        else:                     # original 9-column layout
            n_s, nb_s = r[3], r[4]
        gpus = re.match(r"\s*(\d+)", r[1])
        gpus = int(gpus.group(1)) if gpus else None
        g, n, nb = first_num(r[2]), first_num(n_s), first_num(nb_s)
        note = r[-1].lower()
        if r[0].strip() != "A100_80" or gpus is None or gpus < min_gpus:
            continue
        if None in (g, n, nb) or "cut off" in note:
            continue
        recs.append({"N": int(n), "NB": int(nb), "GPUs": gpus, "Gflops": g})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("-o", "--output", default="hpl_score_heatmap.png")
    ap.add_argument("--min-gpus", type=int, default=220)
    a = ap.parse_args()

    df = load(a.csv_file, a.min_gpus)
    # Keep the best run per (N, NB); GPU count is carried along for the row label.
    best = df.sort_values("Gflops", ascending=False).drop_duplicates(["N", "NB"])
    best = best.sort_values(["N", "NB"])
    best["row"] = best["N"].astype(str) + " (" + best["GPUs"].astype(str) + ")"
    order = best.drop_duplicates("N")["row"].tolist()
    pivot = best.pivot_table(index="row", columns="NB", values="Gflops", aggfunc="max")
    pivot = pivot.reindex(order)

    plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Gflops"},
    )
    plt.title("HPL Performance Heatmap (Gflops vs N and NB)", fontsize=14)
    plt.xlabel("NB (Block Size)")
    plt.ylabel("N (Problem Size), GPU count in parentheses")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    ax = plt.gca()
    ax.grid(which="major", color="gray", linestyle="-", linewidth=0.5)
    ax.grid(which="minor", color="lightgray", linestyle="--", linewidth=0.3)
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(a.output, dpi=300)
    plt.close()

    print(f"{len(df)} runs -> {len(best)} unique (N, NB) cells; saved {a.output}")
    print(pivot.to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()
