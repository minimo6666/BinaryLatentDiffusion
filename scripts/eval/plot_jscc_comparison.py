"""Add AdaLN ablation results to existing JSCC comparison plot.
Does NOT modify original plots — produces a new combined figure.

Usage:
    python scripts/eval/plot_jscc_comparison.py
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 路径配置
# ==========================================
base_dir = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/experiments"
diffusion_file = os.path.join(base_dir, "diffusion_different_snr_performance/diffusion_psnr_summary.txt")
jscc_base_dir = os.path.join(base_dir, "jscc_different_snr_performance")
adaln_metrics_dir = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln/metrics_results/16QAM"
pdf_save_dir = os.path.join(base_dir, "pdf")

os.makedirs(pdf_save_dir, exist_ok=True)

# JSCC 训练信噪比点
jscc_train_snrs = [-15, 0, 4, 8, 25]

# AdaLN 只取这几个 SNR 点
adaln_snr_targets = [0, 4, 8, 12, 15]


# ==========================================
# 2. 解析函数
# ==========================================
def parse_txt_universal(file_path):
    x_data, y_data = [], []
    if not os.path.exists(file_path):
        print(f"Not found: {file_path}")
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
        indices = np.argsort(x_data)
        return np.array(x_data)[indices], np.array(y_data)[indices]
    return np.array([]), np.array([])


def read_our_metrics(file_path, target_snrs):
    """Read metrics and pick only points closest to target_snrs."""
    snr_vals, metric_vals = [], []
    if not os.path.exists(file_path):
        return np.array([]), np.array([])
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) >= 2:
                snr_vals.append(float(parts[0].replace('_db', '')))
                metric_vals.append(float(parts[1].split(' +/- ')[0]))
    snr_arr = np.array(snr_vals)
    metric_arr = np.array(metric_vals)

    # Pick closest match for each target SNR
    picked_snrs, picked_metrics = [], []
    for target in target_snrs:
        idx = np.argmin(np.abs(snr_arr - target))
        picked_snrs.append(snr_arr[idx])
        picked_metrics.append(metric_arr[idx])
    return np.array(picked_snrs), np.array(picked_metrics)


# ==========================================
# 3. 全局样式
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.family': 'serif'
})

# ==========================================
# 4. PSNR 图 (原图 + AdaLN)
# ==========================================
fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

jscc_colors = ['#9467bd', '#17becf', '#2ca02c', '#ff7f0e', '#7f7f7f']
jscc_markers = ['v', '^', '<', '>', 's']

# 原有 JSCC
for idx, train_snr in enumerate(jscc_train_snrs):
    path = os.path.join(jscc_base_dir, f"fix_snr_{train_snr}", "psnr_summary.txt")
    x, y = parse_txt_universal(path)
    if len(x) > 0:
        ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                markersize=8, alpha=0.6, color=jscc_colors[idx], label=f'JSCC: {train_snr}dB')

# 原有 Ours (原始 diffusion baseline)
diff_x, diff_y = parse_txt_universal(diffusion_file)
if len(diff_x) > 0:
    print(f"Original Ours: SNR {diff_x} -> PSNR {diff_y}")
    ax.plot(diff_x, diff_y, linestyle='-', linewidth=4, marker='o',
            markersize=10, color='#d62728', markeredgecolor='black',
            label='BLD_DSC (token)', zorder=10)

# 新增 AdaLN
adaln_x, adaln_y = read_our_metrics(
    os.path.join(adaln_metrics_dir, "denoise", "psnr", "psnr.txt"),
    adaln_snr_targets)
if len(adaln_x) > 0:
    print(f"AdaLN PSNR: SNR {adaln_x} -> {adaln_y}")
    ax.plot(adaln_x, adaln_y, linestyle='-', linewidth=4, marker='D',
            markersize=10, color='#1f77b4', markeredgecolor='black',
            label='BLD_DSC (AdaLN)', zorder=11)

ax.set_title('PSNR Performance across Channel SNR Range')
ax.set_xlabel('Test Channel SNR (dB)')
ax.set_ylabel('Average PSNR (dB)')
ax.set_xlim([-17, 27])
ax.set_xticks(np.arange(-15, 26, 5))
ax.set_ylim([7, 35])
ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
save_path = os.path.join(pdf_save_dir, "JSCC_vs_Ours_with_AdaLN_psnr.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

# ==========================================
# 5. SSIM 图 (原图 + AdaLN)
# ==========================================
fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

for idx, train_snr in enumerate(jscc_train_snrs):
    path = os.path.join(jscc_base_dir, f"fix_snr_{train_snr}", "ssim_summary.txt")
    x, y = parse_txt_universal(path)
    if len(x) > 0:
        ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                markersize=8, alpha=0.6, color=jscc_colors[idx], label=f'JSCC: {train_snr}dB')

# 原有 Ours SSIM — skip if no file (use our metrics format if available)
adaln_x, adaln_y = read_our_metrics(
    os.path.join(adaln_metrics_dir, "denoise", "ssim", "ssim.txt"),
    adaln_snr_targets)
if len(adaln_x) > 0:
    print(f"AdaLN SSIM: SNR {adaln_x} -> {adaln_y}")
    ax.plot(adaln_x, adaln_y, linestyle='-', linewidth=4, marker='D',
            markersize=10, color='#1f77b4', markeredgecolor='black',
            label='BLD_DSC (AdaLN)', zorder=11)

ax.set_title('SSIM Performance across Channel SNR Range')
ax.set_xlabel('Test Channel SNR (dB)')
ax.set_ylabel('Average SSIM')
ax.set_xlim([-17, 27])
ax.set_xticks(np.arange(-15, 26, 5))
ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
save_path = os.path.join(pdf_save_dir, "JSCC_vs_Ours_with_AdaLN_ssim.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")
