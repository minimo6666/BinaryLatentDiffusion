#!/usr/bin/env python3
"""Aggregate every SNR-specific JSCC baseline and its BLD extension."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DEFAULT_RESULTS_DIR


BASE_SNRS = [-15, 0, 4, 8, 25]
TEST_SNRS = [-15, 0, 4, 8, 12, 15, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def metrics_path(results_dir: Path, base_snr: int) -> Path:
    if base_snr == 25:
        return results_dir / "metrics" / "paired_metrics.csv"
    return results_dir / f"jscc_train_snr_{base_snr}" / "metrics" / "paired_metrics.csv"


def load_metrics(path: Path) -> dict[float, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment metrics: {path}")
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[float(row["test_ebn0_db"])] = {
                key: float(value) for key, value in row.items()
            }
    return result


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        base_snr: load_metrics(metrics_path(results_dir, base_snr))
        for base_snr in BASE_SNRS
    }

    long_rows = []
    for base_snr in BASE_SNRS:
        for test_snr in TEST_SNRS:
            row = all_metrics[base_snr].get(float(test_snr))
            if row is None:
                continue
            long_rows.append(
                {
                    "jscc_train_snr_db": base_snr,
                    "test_ebn0_db": test_snr,
                    "jscc_psnr_db": row["jscc25_psnr_mean"],
                    "jscc_bld_psnr_db": row["jscc25_bld_psnr_mean"],
                    "psnr_gain_db": row["psnr_gain_db"],
                    "jscc_ssim": row["jscc25_ssim_mean"],
                    "jscc_bld_ssim": row["jscc25_bld_ssim_mean"],
                    "jscc_ber": row["jscc25_ber"],
                    "jscc_bld_ber": row["jscc25_bld_ber"],
                }
            )
    long_path = output_dir / "all_jscc_bld_results.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    psnr_rows, psnr_labels, gain_rows, gain_labels = [], [], [], []
    for base_snr in BASE_SNRS:
        baseline_values, bld_values, gain_values = [], [], []
        for test_snr in TEST_SNRS:
            row = all_metrics[base_snr].get(float(test_snr))
            baseline_values.append(None if row is None else row["jscc25_psnr_mean"])
            bld_values.append(None if row is None else row["jscc25_bld_psnr_mean"])
            gain_values.append(None if row is None else row["psnr_gain_db"])
        psnr_labels.extend([f"JSCC {base_snr}", f"JSCC {base_snr} + BLD"])
        psnr_rows.extend(
            [[fmt(value) for value in baseline_values], [fmt(value) for value in bld_values]]
        )
        gain_labels.append(f"JSCC {base_snr} gain")
        gain_rows.append([fmt(value) for value in gain_values])

    columns = [str(value) for value in TEST_SNRS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(14, 10.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)
    top = figure.add_subplot(grid[0])
    top.axis("off")
    top.set_title("Average PSNR (dB): SNR-specific DeepJSCC with and without BLD", pad=14)
    table = top.table(
        cellText=psnr_rows,
        rowLabels=psnr_labels,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)

    bottom = figure.add_subplot(grid[1])
    bottom.axis("off")
    bottom.set_title("PSNR gain from BLD (dB)", pad=14)
    gain_table = bottom.table(
        cellText=gain_rows,
        rowLabels=gain_labels,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    gain_table.auto_set_font_size(False)
    gain_table.set_fontsize(9.5)
    gain_table.scale(1.0, 1.55)
    figure.text(0.5, 0.02, "Columns are test channel Eb/N0 (dB).", ha="center", fontsize=9)

    pdf_path = output_dir / "all_jscc_bld_psnr_table.pdf"
    png_path = output_dir / "all_jscc_bld_psnr_table.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    md_path = output_dir / "all_jscc_bld_psnr_table.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Average PSNR (dB)\n\n")
        handle.write("| Model | " + " | ".join(columns) + " |\n")
        handle.write("|---|" + "---:|" * len(columns) + "\n")
        for label, values in zip(psnr_labels, psnr_rows):
            handle.write(f"| {label} | " + " | ".join(values) + " |\n")
        handle.write("\n# PSNR gain from BLD (dB)\n\n")
        handle.write("| Base model | " + " | ".join(columns) + " |\n")
        handle.write("|---|" + "---:|" * len(columns) + "\n")
        for label, values in zip(gain_labels, gain_rows):
            handle.write(f"| {label} | " + " | ".join(values) + " |\n")

    print(f"Saved aggregate CSV: {long_path}")
    print(f"Saved aggregate table: {pdf_path}")
    print(f"Saved aggregate preview: {png_path}")


if __name__ == "__main__":
    main()
