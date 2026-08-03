#!/usr/bin/env python3
"""Plot only JSCC-25 and JSCC-25 + BLD from paired evaluation metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_RESULTS_DIR / "metrics/paired_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["test_ebn0_db"]))
    x = np.asarray([float(row["test_ebn0_db"]) for row in rows])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
    })

    plots = [
        ("psnr", "Average PSNR (dB)", "PSNR: Frozen JSCC-25 with and without BLD"),
        ("ssim", "Average SSIM", "SSIM: Frozen JSCC-25 with and without BLD"),
        ("ber", "Bit Error Rate", "BER: Frozen JSCC-25 with and without BLD"),
    ]
    for metric, ylabel, title in plots:
        baseline = np.asarray([float(row[f"jscc25_{metric}_mean"] if metric != "ber" else row["jscc25_ber"]) for row in rows])
        ours = np.asarray([float(row[f"jscc25_bld_{metric}_mean"] if metric != "ber" else row["jscc25_bld_ber"]) for row in rows])
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=220)
        ax.plot(x, baseline, "--s", linewidth=2.2, markersize=7, color="#4c78a8", label="Frozen JSCC-25")
        ax.plot(x, ours, "-o", linewidth=3.2, markersize=8, color="#d62728", markeredgecolor="black", label="Frozen JSCC-25 + BLD")
        if metric == "ber":
            ax.set_yscale("log")
            positive = np.concatenate([baseline[baseline > 0], ours[ours > 0]])
            if positive.size:
                ax.set_ylim(max(positive.min() / 2, 1e-8), min(max(positive.max() * 2, 1e-3), 1.0))
        ax.set_xlabel("Test channel Eb/N0 (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(frameon=True)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(args.output_dir / f"jscc25_vs_jscc25_bld_{metric}.{extension}", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
