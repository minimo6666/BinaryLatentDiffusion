#!/usr/bin/env python3
"""Plot BFM SNR/PSNR/BER and the final fair JSCC comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_DIR
from experiments.different_snr_jscc.common import safe_snr_name

TRAIN_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bfm-summary", type=Path, default=EXPERIMENT_DIR / "results" / "summary.csv"
    )
    parser.add_argument(
        "--jscc-results",
        type=Path,
        default=EXPERIMENT_DIR.parent / "different_snr_jscc" / "results",
    )
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
    bfm = read_rows(args.bfm_summary)
    x = np.array([row["eval_ebn0_db"] for row in bfm])
    raw_psnr = np.array([row["raw_qam_psnr_mean_db"] for row in bfm])
    bfm_direct_psnr = np.array([row["bfm_direct_psnr_mean_db"] for row in bfm])
    bfm_iterative_psnr = np.array([row["bfm_iterative_psnr_mean_db"] for row in bfm])
    raw_ber = np.array([row["measured_channel_ber"] for row in bfm])
    bfm_direct_ber = np.array([row["direct_denoised_ber"] for row in bfm])
    bfm_iterative_ber = np.array([row["iterative_denoised_ber"] for row in bfm])
    theory = np.array([row["theoretical_channel_ber"] for row in bfm])

    matched_jscc = {}
    best_jscc = {}
    for train_snr in TRAIN_SNRS:
        path = (
            args.jscc_results
            / f"train_snr_{safe_snr_name(train_snr)}"
            / "summary.csv"
        )
        if not path.is_file():
            continue
        rows = read_rows(path)
        for row in rows:
            eval_snr = row["eval_ebn0_db"]
            value = row["noisy_psnr_mean_db"]
            best_jscc[eval_snr] = max(best_jscc.get(eval_snr, -np.inf), value)
            if np.isclose(eval_snr, train_snr):
                matched_jscc[eval_snr] = value

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].plot(x, raw_psnr, "s--", label="Frozen C64 AE, raw QAM bits")
    axes[0].plot(x, bfm_direct_psnr, "o-", linewidth=2.2, label="BFM direct X0")
    axes[0].plot(x, bfm_iterative_psnr, "v-", linewidth=1.8, label="BFM iterative")
    if matched_jscc:
        jx = np.array(sorted(matched_jscc))
        axes[0].plot(
            jx,
            [matched_jscc[value] for value in jx],
            "D-",
            linewidth=2.0,
            label="JSCC trained/tested at matched SNR",
        )
    if best_jscc:
        jx = np.array(sorted(best_jscc))
        axes[0].plot(
            jx,
            [best_jscc[value] for value in jx],
            "^:",
            linewidth=1.8,
            label="Best fixed-SNR JSCC at each test SNR",
        )
    axes[0].set_xlabel("Eb/N0 (dB)")
    axes[0].set_ylabel("PSNR on shared 100 FFHQ test images (dB)")
    axes[0].set_title("Fair C64 / 16,384-bit comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].semilogy(x, theory, "k--", label="Gray 16-QAM theory")
    axes[1].semilogy(x, raw_ber, "s-", label="Measured channel BER")
    axes[1].semilogy(x, bfm_direct_ber, "o-", label="BFM direct-X0 BER")
    axes[1].semilogy(x, bfm_iterative_ber, "v-", label="BFM iterative BER")
    axes[1].set_xlabel("Eb/N0 (dB)")
    axes[1].set_ylabel("BER")
    axes[1].set_title("BFM bit correction")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fair_jscc_bfm_snr_psnr_ber.png", dpi=240)
    fig.savefig(args.output_dir / "fair_jscc_bfm_snr_psnr_ber.pdf")
    plt.close(fig)

    comparison_rows = []
    for index, snr in enumerate(x):
        comparison_rows.append(
            {
                **bfm[index],
                "matched_jscc_psnr_db": matched_jscc.get(float(snr), float("nan")),
                "best_jscc_psnr_db": best_jscc.get(float(snr), float("nan")),
            }
        )
    with (args.output_dir / "fair_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved final comparison under {args.output_dir}")


if __name__ == "__main__":
    main()
