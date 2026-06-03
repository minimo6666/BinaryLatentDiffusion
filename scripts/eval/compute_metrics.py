"""Compute PSNR and SSIM for denoised and noisy images versus GT.

Usage:
    python scripts/eval/compute_metrics.py \
        --data_dir /path/to/samples_for_psnr/16QAM \
        --gt_dir /path/to/samples_for_psnr/16QAM/gt_100000 \
        --result_dir /path/to/metrics_results/16QAM \
        --qam 16QAM
"""
import os
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_metrics_for_mode(data_dir, gt_dir, mode="denoise"):
    """Compute PSNR/SSIM for all SNR folders under data_dir/mode."""
    mode_path = os.path.join(data_dir, mode)
    if not os.path.exists(mode_path):
        print(f"Warning: {mode} directory not found at {mode_path}")
        return {}, {}

    db_folders = sorted(
        [f for f in os.listdir(mode_path) if f.endswith('_db')],
        key=lambda x: float(x.split('_')[0])
    )

    psnr_results = {}
    ssim_results = {}

    for db_folder in db_folders:
        db_path = os.path.join(mode_path, db_folder)
        psnr_vals = []
        ssim_vals = []

        for img_name in sorted(os.listdir(db_path)):
            if not img_name.endswith('.png'):
                continue

            db_img_path = os.path.join(db_path, img_name)
            # GT uses the same index number
            gt_name = f"{img_name.split('.')[0]}.png"
            gt_path = os.path.join(gt_dir, gt_name)

            if not os.path.exists(gt_path):
                print(f"  Warning: GT {gt_name} not found for {img_name}")
                continue

            db_img = cv2.imread(db_img_path)
            gt_img = cv2.imread(gt_path)

            if db_img is None or gt_img is None:
                print(f"  Warning: failed to read {img_name} or GT")
                continue

            try:
                p_val = psnr(gt_img, db_img, data_range=255)
                s_val = ssim(gt_img, db_img, channel_axis=2, win_size=11, data_range=255)
                psnr_vals.append(p_val)
                ssim_vals.append(s_val)
            except Exception as e:
                print(f"  Error computing metrics for {img_name}: {e}")
                continue

        if psnr_vals:
            snr_key = float(db_folder.split('_')[0])
            psnr_results[snr_key] = (np.mean(psnr_vals), np.std(psnr_vals))
            ssim_results[snr_key] = (np.mean(ssim_vals), np.std(ssim_vals))
            print(f"  [{mode}] SNR={snr_key:.2f} dB: PSNR={psnr_results[snr_key][0]:.4f}, SSIM={ssim_results[snr_key][0]:.4f}")
        else:
            print(f"  [{mode}] SNR={db_folder}: no valid images")

    return psnr_results, ssim_results


def save_metrics(results, result_dir, metric_name):
    """Save metrics to text file."""
    os.makedirs(os.path.join(result_dir, metric_name), exist_ok=True)
    filepath = os.path.join(result_dir, metric_name, f"{metric_name}.txt")
    with open(filepath, "w") as f:
        for snr in sorted(results.keys()):
            mean_val, std_val = results[snr]
            f.write(f"{snr:.2f}_db: {mean_val:.4f} +/- {std_val:.4f}\n")
    print(f"Saved {metric_name} to {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to samples_for_psnr/16QAM (contains noise/ and denoise/ subdirs)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Path to GT images directory")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to save metrics_results")
    args = parser.parse_args()

    for mode in ["denoise", "noise"]:
        print(f"\nComputing metrics for [{mode}]...")
        psnr_res, ssim_res = compute_metrics_for_mode(args.data_dir, args.gt_dir, mode=mode)

        if psnr_res:
            mode_result_dir = os.path.join(args.result_dir, mode)
            os.makedirs(mode_result_dir, exist_ok=True)
            save_metrics(psnr_res, mode_result_dir, "psnr")
            save_metrics(ssim_res, mode_result_dir, "ssim")

    print("\nDone.")


if __name__ == "__main__":
    main()
