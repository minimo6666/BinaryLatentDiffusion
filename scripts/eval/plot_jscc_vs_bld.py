"""
Compare JSCC (autoencoder-only) vs BLD_DSC (diffusion denoising) PSNR.

Usage (standalone):
    python scripts/eval/plot_jscc_vs_bld.py \
        --jscc_base /mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq \
        --bld_metrics /path/to/bld/metrics_results/16QAM \
        --save_path ./jscc_vs_ours.pdf
"""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_jscc_summary(file_path):
    """Parse the comma-separated psnr_summary.txt produced by gen_imgs_same_jscc_under_diff_snr.py.

    Format:
        === PSNR Evaluation Summary ===
        Test_SNR(dB), Average_PSNR(dB)
        -----------------------------------
                 -15,          15.4228
                   0,          16.0566
    """
    x_data, y_data = [], []
    if not os.path.exists(file_path):
        print(f"  [WARN] JSCC file not found: {file_path}")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('Test_SNR') or set(line.strip()) == {'-'}:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    x_data.append(float(parts[0].strip()))
                    y_data.append(float(parts[1].strip()))
                except ValueError:
                    continue

    if x_data:
        idx = np.argsort(x_data)
        return np.array(x_data)[idx], np.array(y_data)[idx]
    return np.array([]), np.array([])


def read_bld_metrics(file_path):
    """Read BLD_DSC-format metrics (e.g. '0.00_db: 19.8949 +/- 2.2748')."""
    snr_vals, metric_vals = [], []
    if not os.path.exists(file_path):
        print(f"  [WARN] BLD metrics not found: {file_path}")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(': ')
            if len(parts) >= 2:
                snr = float(parts[0].replace('_db', ''))
                val = float(parts[1].split(' +/- ')[0])
                snr_vals.append(snr)
                metric_vals.append(val)

    idx = np.argsort(snr_vals)
    return np.array(snr_vals)[idx], np.array(metric_vals)[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jscc_base", type=str, required=True,
                        help="Base dir containing en_de_train_under_snr_{snr}/ dirs")
    parser.add_argument("--bld_metrics", type=str, required=True,
                        help="Path to BLD_DSC metrics_results/16QAM dir")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Full path for output PDF (e.g. /path/to/JSCC_vs_Ours_Semantic_1000_PSNR.pdf)")
    parser.add_argument("--bld_label", type=str, default="BLD_DSC (Semantic Loss)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    jscc_train_snrs = [-15, 0, 4, 8, 25]
    jscc_colors = ['#9467bd', '#17becf', '#2ca02c', '#ff7f0e', '#7f7f7f']
    jscc_markers = ['v', '^', '<', '>', 's']

    # Global style
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'font.family': 'serif',
    })

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    # ---- JSCC baselines ----
    for idx, train_snr in enumerate(jscc_train_snrs):
        path = os.path.join(args.jscc_base, f"en_de_train_under_snr_{train_snr}", "psnr_summary.txt")
        x, y = parse_jscc_summary(path)
        if len(x) > 0:
            ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                    markersize=8, alpha=0.6, color=jscc_colors[idx],
                    label=f'JSCC: {train_snr}dB')
            print(f"  JSCC {train_snr}dB: {len(x)} points, PSNR range [{y.min():.2f}, {y.max():.2f}]")

    # ---- BLD_DSC (ours) ----
    bld_psnr_path = os.path.join(args.bld_metrics, "denoise", "psnr", "psnr.txt")
    bld_x, bld_y = read_bld_metrics(bld_psnr_path)
    if len(bld_x) > 0:
        ax.plot(bld_x, bld_y, linestyle='-', linewidth=4, marker='o',
                markersize=10, color='#d62728', markeredgecolor='black',
                label=args.bld_label, zorder=10)
        print(f"  BLD_DSC: {len(bld_x)} points, PSNR range [{bld_y.min():.2f}, {bld_y.max():.2f}]")

    ax.set_title('PSNR Performance across Channel SNR Range')
    ax.set_xlabel('Test Channel SNR (dB)')
    ax.set_ylabel('Average PSNR (dB)')
    ax.set_xlim([-17, 27])
    ax.set_xticks(np.arange(-15, 26, 5))
    ax.set_ylim([5, 35])
    ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(args.save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()
