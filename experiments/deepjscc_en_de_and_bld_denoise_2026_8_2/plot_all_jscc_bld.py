#!/usr/bin/env python3
"""Plot all five SNR-specific DeepJSCC/BLD comparisons in one figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DEFAULT_RESULTS_DIR


BASE_SNRS = [-15, 0, 4, 8, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def metrics_path(results_dir: Path, base_snr: int) -> Path:
    if base_snr == 25:
        return results_dir / "metrics" / "paired_metrics.csv"
    return results_dir / f"jscc_train_snr_{base_snr}" / "metrics" / "paired_metrics.csv"


def load_curve(path: Path):
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
    return list(zip(*rows))


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    all_values = []

    for axis, base_snr in zip(axes, BASE_SNRS):
        test_snr, baseline, bld = load_curve(metrics_path(results_dir, base_snr))
        all_values.extend(baseline)
        all_values.extend(bld)
        axis.plot(
            test_snr,
            baseline,
            color="#4C78A8",
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=5.5,
            label=f"Frozen JSCC-{base_snr}",
        )
        axis.plot(
            test_snr,
            bld,
            color="#E02626",
            linestyle="-",
            linewidth=2.4,
            marker="o",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=5.5,
            label=f"Frozen JSCC-{base_snr} + BLD",
        )
        axis.set_title(f"DeepJSCC trained at {base_snr} dB")
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.legend(loc="best", frameon=True)
        axis.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20, 25])
        axis.set_xlim(-17, 27)

    axes[-1].axis("off")
    margin = max(0.5, (max(all_values) - min(all_values)) * 0.04)
    for axis in axes[:-1]:
        axis.set_ylim(min(all_values) - margin, max(all_values) + margin)
    for axis in axes[3:5]:
        axis.set_xlabel("Test channel Eb/N0 (dB)")
    axes[0].set_ylabel("Average PSNR (dB)")
    axes[3].set_ylabel("Average PSNR (dB)")
    figure.suptitle(
        "PSNR: SNR-specific frozen DeepJSCC with and without BLD",
        fontsize=16,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))

    pdf_path = output_dir / "all_jscc_bld_psnr_comparison.pdf"
    png_path = output_dir / "all_jscc_bld_psnr_comparison.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved combined PSNR figure: {pdf_path}")
    print(f"Saved combined PSNR preview: {png_path}")


if __name__ == "__main__":
    main()
