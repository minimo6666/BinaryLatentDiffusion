"""Plot PSNR/SSIM comparison curves for denoised vs noisy vs DeepJSCC baselines.

Usage:
    python scripts/eval/plot_metrics.py \
        --result_dir /path/to/metrics_results/16QAM \
        --save_dir /path/to/pdf_output \
        --label "BLD_DSC (AdaLN)"
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def read_metrics(file_path):
    """Read metrics from format: 'SNR_db: mean +/- std'."""
    snr_vals = []
    metric_vals = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(': ')
            if len(parts) >= 2:
                snr = float(parts[0].replace('_db', ''))
                val_str = parts[1].split(' +/- ')[0]
                metric = float(val_str)
                snr_vals.append(snr)
                metric_vals.append(metric)
    idx = np.argsort(snr_vals)
    return np.array(snr_vals)[idx], np.array(metric_vals)[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to metrics_results/16QAM (contains denoise/ and noise/ subdirs)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save PDF plots")
    parser.add_argument("--label", type=str, default="BLD_DSC",
                        help="Label for the experiment")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Read data
    denoise_psnr_snr, denoise_psnr = read_metrics(
        os.path.join(args.result_dir, "denoise", "psnr", "psnr.txt"))
    denoise_ssim_snr, denoise_ssim = read_metrics(
        os.path.join(args.result_dir, "denoise", "ssim", "ssim.txt"))
    noise_psnr_snr, noise_psnr = read_metrics(
        os.path.join(args.result_dir, "noise", "psnr", "psnr.txt"))
    noise_ssim_snr, noise_ssim = read_metrics(
        os.path.join(args.result_dir, "noise", "ssim", "ssim.txt"))

    # ---- PSNR comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(denoise_psnr_snr, denoise_psnr, 'b-o', linewidth=1.5, markersize=5,
             label=f'{args.label} (denoised)')
    plt.plot(noise_psnr_snr, noise_psnr, 'r-s', linewidth=1.5, markersize=5,
             label=f'{args.label} (noisy)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'PSNR vs SNR — {args.label}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psnr_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- SSIM comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(denoise_ssim_snr, denoise_ssim, 'b-o', linewidth=1.5, markersize=5,
             label=f'{args.label} (denoised)')
    plt.plot(noise_ssim_snr, noise_ssim, 'r-s', linewidth=1.5, markersize=5,
             label=f'{args.label} (noisy)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title(f'SSIM vs SNR — {args.label}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "ssim_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Combined denoised PSNR + SSIM ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(denoise_psnr_snr, denoise_psnr, 'b-o', linewidth=1.5, markersize=5)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title(f'{args.label} — Denoised PSNR')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(denoise_ssim_snr, denoise_ssim, 'r-s', linewidth=1.5, markersize=5)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'{args.label} — Denoised SSIM')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "denoise_metrics.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {args.save_dir}")


if __name__ == "__main__":
    main()
