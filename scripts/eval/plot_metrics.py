import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def read_metrics(file_path):
    snr_vals, metric_vals = [], []
    if not os.path.exists(file_path): return np.array([]), np.array([])
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
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
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="BLD_DSC")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 读取现有指标
    denoise_psnr_snr, denoise_psnr = read_metrics(os.path.join(args.result_dir, "denoise", "psnr", "psnr.txt"))
    noise_psnr_snr, noise_psnr = read_metrics(os.path.join(args.result_dir, "noise", "psnr", "psnr.txt"))

    # 🌟 新增：读取 BER
    ber_noisy_snr, ber_noisy = read_metrics(os.path.join(args.result_dir, "ber", "noisy_ber.txt"))
    ber_denoise_snr, ber_denoise = read_metrics(os.path.join(args.result_dir, "ber", "denoise_ber.txt"))

    # 画图逻辑 (PSNR + BER)
    plt.figure(figsize=(8, 5))
    plt.plot(denoise_psnr_snr, denoise_psnr, 'b-o', label=f'{args.label} (denoised)')
    plt.plot(noise_psnr_snr, noise_psnr, 'r-s', label=f'{args.label} (noisy)')
    plt.ylabel('PSNR (dB)')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(args.save_dir, "psnr_comparison.pdf"))

    # 🌟 新增：对数坐标 BER 图
    plt.figure(figsize=(8, 5))
    plt.semilogy(ber_noisy_snr, ber_noisy, 'r--s', label='Noisy BER')
    plt.semilogy(ber_denoise_snr, ber_denoise, 'b-o', label='Denoised BER')
    plt.ylabel('Bit Error Rate')
    plt.yscale('log')
    plt.legend(); plt.grid(True, which="both"); plt.savefig(os.path.join(args.save_dir, "ber_comparison.pdf"))

if __name__ == "__main__":
    main()