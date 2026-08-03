"""
Full comparison plot: DeepJSCC baselines + BLD_DSC + BLD_DSC (AdaLN).

Adds the AdaLN-Diffusion model's PSNR curve to the existing JSCC comparison,
so all three families appear on one figure.

Usage:
    python scripts/eval/plot_metrics_full.py \
        --result_dir /path/to/your_model/metrics_results/16QAM \
        --jscc_dir /path/to/jscc_different_snr_performance \
        --save_dir ./scripts/eval/results/pdf
"""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 0. 路径常量
# ==========================================
ADALN_METRICS_DIR = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln/metrics_results/16QAM"


# ==========================================
# 1. 解析函数
# ==========================================
def read_metrics(file_path, target_snrs=None):
    """Read BLD_DSC-format metrics (e.g. '0.00_db: 19.8949 +/- 2.2748').

    If target_snrs is provided, only pick the closest-matching SNR points.
    """
    snr_vals, metric_vals = [], []
    if not os.path.exists(file_path):
        print(f"  [WARN] Not found: {file_path}")
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

    snr_arr = np.array(snr_vals)
    metric_arr = np.array(metric_vals)

    if target_snrs is not None:
        picked_snrs, picked_metrics = [], []
        for target in target_snrs:
            idx = np.argmin(np.abs(snr_arr - target))
            picked_snrs.append(snr_arr[idx])
            picked_metrics.append(metric_arr[idx])
        return np.array(picked_snrs), np.array(picked_metrics)

    idx = np.argsort(snr_arr)
    return snr_arr[idx], metric_arr[idx]


def parse_jscc_txt(file_path):
    """Parse DeepJSCC psnr_summary.txt files."""
    x_data, y_data = [], []
    if not os.path.exists(file_path):
        print(f"  [WARN] JSCC file not found: {file_path}")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().replace(',', ' ')
            if not line or any(k in line for k in ['=', 'SNR', 'Target']):
                continue
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) >= 2:
                x_data.append(nums[0])
                y_data.append(nums[-1])

    if x_data:
        idx = np.argsort(x_data)
        return np.array(x_data)[idx], np.array(y_data)[idx]
    return np.array([]), np.array([])


# ==========================================
# 2. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to your model's metrics_results/16QAM directory")
    parser.add_argument("--jscc_dir", type=str, default=None,
                        help="Path to jscc_different_snr_performance directory")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save output PDFs")
    parser.add_argument("--label", type=str, default="BLD_DSC (Semantic Loss)")
    parser.add_argument("--adaln_label", type=str, default="BLD_DSC (AdaLN)")
    parser.add_argument("--adaln_metrics_dir", type=str, default=ADALN_METRICS_DIR,
                        help="Path to AdaLN model's metrics_results/16QAM directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # JSCC 训练信噪比点
    jscc_train_snrs = [-15, 0, 4, 8, 25]
    # AdaLN 已评估的 SNR 点 (根据 psnr.txt 中的实际数据)
    adaln_snr_targets = [0, 4, 8, 12, 15]

    # ==========================================
    # 3. 读取数据
    # ==========================================
    print("=" * 60)
    print("Reading metrics...")

    # 当前模型
    denoise_psnr_snr, denoise_psnr = read_metrics(
        os.path.join(args.result_dir, "denoise", "psnr", "psnr.txt"))
    print(f"  [{args.label}] denoised PSNR: {len(denoise_psnr_snr)} points")

    # AdaLN 模型
    adaln_x, adaln_y = read_metrics(
        os.path.join(args.adaln_metrics_dir, "denoise", "psnr", "psnr.txt"),
        target_snrs=adaln_snr_targets)
    if len(adaln_x) > 0:
        print(f"  [{args.adaln_label}] PSNR: {list(zip(adaln_x, adaln_y))}")
    else:
        print(f"  [WARN] No AdaLN data found — skipping AdaLN curve.")

    # ==========================================
    # 4. 全局样式
    # ==========================================
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
    })

    # ==========================================
    # 5. PSNR 全量对比图
    # ==========================================
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    jscc_colors = ['#9467bd', '#17becf', '#2ca02c', '#ff7f0e', '#7f7f7f']
    jscc_markers = ['v', '^', '<', '>', 's']

    # --- JSCC baselines ---
    if args.jscc_dir and os.path.exists(args.jscc_dir):
        for idx, train_snr in enumerate(jscc_train_snrs):
            path = os.path.join(args.jscc_dir, f"fix_snr_{train_snr}", "psnr_summary.txt")
            x, y = parse_jscc_txt(path)
            if len(x) > 0:
                ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                        markersize=8, alpha=0.6, color=jscc_colors[idx],
                        label=f'JSCC: {train_snr}dB')
                print(f"  JSCC {train_snr}dB: {len(x)} points")
    else:
        print("  [INFO] No JSCC dir provided — skipping JSCC baselines.")

    # --- 当前模型 (ours) ---
    if len(denoise_psnr_snr) > 0:
        ax.plot(denoise_psnr_snr, denoise_psnr,
                linestyle='-', linewidth=4, marker='o',
                markersize=10, color='#d62728', markeredgecolor='black',
                label=args.label, zorder=10)

    # --- AdaLN 模型 ---
    if len(adaln_x) > 0:
        ax.plot(adaln_x, adaln_y,
                linestyle='-', linewidth=4, marker='D',
                markersize=10, color='#1f77b4', markeredgecolor='black',
                label=args.adaln_label, zorder=11)

    ax.set_title('PSNR Performance across Channel SNR Range')
    ax.set_xlabel('Test Channel SNR (dB)')
    ax.set_ylabel('Average PSNR (dB)')
    ax.set_xlim([-17, 27])
    ax.set_xticks(np.arange(-15, 26, 5))
    ax.set_ylim([5, 35])
    ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    pdf_path = os.path.join(args.save_dir, "JSCC_vs_Ours_full_psnr.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {pdf_path}")

    # ==========================================
    # 6. (可选) SSIM 全量对比图
    # ==========================================
    denoise_ssim_snr, denoise_ssim = read_metrics(
        os.path.join(args.result_dir, "denoise", "ssim", "ssim.txt"))

    adaln_ssim_x, adaln_ssim_y = read_metrics(
        os.path.join(args.adaln_metrics_dir, "denoise", "ssim", "ssim.txt"),
        target_snrs=adaln_snr_targets)

    has_ssim = len(denoise_ssim_snr) > 0 or len(adaln_ssim_x) > 0

    if has_ssim:
        fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

        if args.jscc_dir and os.path.exists(args.jscc_dir):
            for idx, train_snr in enumerate(jscc_train_snrs):
                path = os.path.join(args.jscc_dir, f"fix_snr_{train_snr}", "ssim_summary.txt")
                x, y = parse_jscc_txt(path)
                if len(x) > 0:
                    ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                            markersize=8, alpha=0.6, color=jscc_colors[idx],
                            label=f'JSCC: {train_snr}dB')

        if len(denoise_ssim_snr) > 0:
            ax.plot(denoise_ssim_snr, denoise_ssim,
                    linestyle='-', linewidth=4, marker='o',
                    markersize=10, color='#d62728', markeredgecolor='black',
                    label=args.label, zorder=10)

        if len(adaln_ssim_x) > 0:
            ax.plot(adaln_ssim_x, adaln_ssim_y,
                    linestyle='-', linewidth=4, marker='D',
                    markersize=10, color='#1f77b4', markeredgecolor='black',
                    label=args.adaln_label, zorder=11)

        ax.set_title('SSIM Performance across Channel SNR Range')
        ax.set_xlabel('Test Channel SNR (dB)')
        ax.set_ylabel('Average SSIM')
        ax.set_xlim([-17, 27])
        ax.set_xticks(np.arange(-15, 26, 5))
        ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        ssim_path = os.path.join(args.save_dir, "JSCC_vs_Ours_full_ssim.pdf")
        plt.savefig(ssim_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {ssim_path}")

    print("Done.")


if __name__ == "__main__":
    main()
