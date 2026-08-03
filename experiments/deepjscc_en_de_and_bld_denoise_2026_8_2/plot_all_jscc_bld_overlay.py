#!/usr/bin/env python3
"""Overlay all SNR-specific DeepJSCC baselines and BLD curves on one axis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_RESULTS_DIR


BASE_SNRS = [-15, 0, 4, 8, 25]
COLORS = ["#9467bd", "#17becf", "#2ca02c", "#ff7f0e", "#7f7f7f"]
MARKERS = ["v", "^", "<", ">", "s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def metrics_path(results_dir: Path, base_snr: int) -> Path:
    if base_snr == 25:
        return results_dir / "metrics" / "paired_metrics.csv"
    return results_dir / f"jscc_train_snr_{base_snr}" / "metrics" / "paired_metrics.csv"


def read_curve(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    float(row["test_ebn0_db"]),
                    float(row["jscc25_psnr_mean"]),
                    float(row["jscc25_bld_psnr_mean"]),
                )
            )
    rows.sort(key=lambda item: item[0])
    return tuple(np.asarray(values) for values in zip(*rows))


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 10,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(12, 9), dpi=300)

    for index, base_snr in enumerate(BASE_SNRS):
        test_snr, baseline, bld = read_curve(metrics_path(results_dir, base_snr))
        color = COLORS[index]
        marker = MARKERS[index]
        axis.plot(
            test_snr,
            baseline,
            linestyle="--",
            linewidth=1.5,
            marker=marker,
            markersize=8,
            alpha=0.58,
            color=color,
            label=f"JSCC: {base_snr}dB",
            zorder=2,
        )
        axis.plot(
            test_snr,
            bld,
            linestyle="-",
            linewidth=3.0,
            marker=marker,
            markersize=8,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"JSCC: {base_snr}dB + BLD",
            zorder=8,
        )

    axis.set_title("PSNR Performance across Channel SNR Range")
    axis.set_xlabel("Test Channel SNR (dB)")
    axis.set_ylabel("Average PSNR (dB)")
    axis.set_xlim([-17, 27])
    axis.set_xticks(np.arange(-15, 26, 5))
    axis.set_ylim([5, 35])
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        shadow=True,
        edgecolor="black",
        columnspacing=1.2,
        handlelength=2.8,
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    figure.tight_layout()

    pdf_path = output_dir / "all_jscc_bld_psnr_overlay.pdf"
    png_path = output_dir / "all_jscc_bld_psnr_overlay.png"
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    figure.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved overlay PDF: {pdf_path}")
    print(f"Saved overlay PNG: {png_path}")


if __name__ == "__main__":
    main()
