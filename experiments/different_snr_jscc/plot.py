#!/usr/bin/env python3
"""Aggregate fixed-training-SNR JSCC evaluations and plot SNR-to-PSNR curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_DIR

TRAIN_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "plots")
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    by_train = {}
    for train_snr in TRAIN_SNRS:
        name = f"{train_snr:g}".replace("-", "minus").replace(".", "p")
        path = args.result_root / f"train_snr_{name}" / "summary.csv"
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        rows = read_rows(path)
        by_train[float(train_snr)] = rows
        all_rows.extend(rows)
    if not by_train:
        raise FileNotFoundError("No JSCC summary.csv files found")

    with (args.output_dir / "all_jscc_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axis = plt.subplots(figsize=(10, 6.5))
    matched_x, matched_y = [], []
    for train_snr, rows in sorted(by_train.items()):
        x = np.array([row["eval_ebn0_db"] for row in rows])
        y = np.array([row["noisy_psnr_mean_db"] for row in rows])
        axis.plot(x, y, marker="o", linewidth=1.6, label=f"trained at {train_snr:g} dB")
        match = np.where(np.isclose(x, train_snr))[0]
        if match.size:
            matched_x.append(train_snr)
            matched_y.append(y[match[0]])
    if matched_x:
        order = np.argsort(matched_x)
        axis.plot(
            np.asarray(matched_x)[order],
            np.asarray(matched_y)[order],
            color="black",
            marker="D",
            linewidth=2.5,
            linestyle="--",
            label="matched train/test SNR",
        )
    axis.set_xlabel("Evaluation Eb/N0 (dB)")
    axis.set_ylabel("PSNR on 100 held-out FFHQ images (dB)")
    axis.set_title("C64 JSCC: fixed-SNR training, cross-SNR evaluation")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "snr_psnr.png", dpi=220)
    fig.savefig(args.output_dir / "snr_psnr.pdf")
    plt.close(fig)

    trains = sorted(by_train)
    evals = sorted({row["eval_ebn0_db"] for rows in by_train.values() for row in rows})
    matrix = np.full((len(trains), len(evals)), np.nan)
    for row_index, train_snr in enumerate(trains):
        lookup = {
            row["eval_ebn0_db"]: row["noisy_psnr_mean_db"]
            for row in by_train[train_snr]
        }
        for column_index, eval_snr in enumerate(evals):
            matrix[row_index, column_index] = lookup.get(eval_snr, np.nan)
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(evals)), [f"{value:g}" for value in evals])
    axis.set_yticks(range(len(trains)), [f"{value:g}" for value in trains])
    axis.set_xlabel("Evaluation Eb/N0 (dB)")
    axis.set_ylabel("Training Eb/N0 (dB)")
    axis.set_title("JSCC PSNR (dB)")
    for row_index in range(len(trains)):
        for column_index in range(len(evals)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row_index, column_index] < np.nanmean(matrix) else "black",
            )
    fig.colorbar(image, ax=axis, label="PSNR (dB)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "psnr_heatmap.png", dpi=220)
    plt.close(fig)

    representative = by_train[sorted(by_train, key=lambda x: abs(x))[0]]
    x = np.array([row["eval_ebn0_db"] for row in representative])
    theory = np.array([row["theoretical_channel_ber"] for row in representative])
    measured = np.array([row["measured_channel_ber"] for row in representative])
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.semilogy(x, theory, "k--", label="exact Gray-QAM theory")
    axis.semilogy(x, measured, "o-", label="measured channel BER")
    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel("BER")
    axis.set_title("Shared physical channel audit")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "ber_channel_audit.png", dpi=220)
    plt.close(fig)
    print(f"Saved plots under {args.output_dir}")


if __name__ == "__main__":
    main()
