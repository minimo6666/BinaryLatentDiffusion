"""
Memory-efficient sampling + metrics + plotting for BLD_DSC QAM experiments.

Usage:
    bash scripts/eval/run_sample_and_eval.sh
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import copy
import argparse

from models.binaryae import BinaryAutoEncoder
from hparams import get_sampler_hparams
from utils.sampler_utils import (
    retrieve_autoencoder_components_state_dicts,
    get_sampler,
    get_online_samples_denoise_code,
    get_online_decode_code_into_images,
)
from utils.log_utils import log, config_log, start_training_log, load_model
import misc
import torch.distributed as dist
from omegaconf import OmegaConf
import glob
from ldm.ldm.util import instantiate_from_config
from utils.qam_utils import *
from utils.m_qam_awgn_util import *
import torchvision
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
    return make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)


def batched_encode(bergan, images, device, batch_size=8):
    """Encode images through BAE encoder in small batches."""
    codes = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            code = bergan(batch, code_only=True).detach()
        codes.append(code.cpu())
    return torch.cat(codes, dim=0)


def batched_decode(bergan, x_codes, H, device, batch_size=8):
    """Decode latent codes to images in small batches."""
    imgs = []
    for i in range(0, x_codes.shape[0], batch_size):
        batch = x_codes[i:i+batch_size].to(device)
        with torch.no_grad():
            img_list = get_online_decode_code_into_images(batch, bergan, H)
        for img in img_list:
            imgs.append(img.cpu())
    return imgs


def batched_denoise(sample_model, x_noisy, noise_t, device, batch_size=8):
    """Run diffusion denoising in small batches."""
    denoised = []
    for i in range(0, x_noisy.shape[0], batch_size):
        batch = x_noisy[i:i+batch_size].to(device)
        with torch.no_grad():
            result = get_online_samples_denoise_code(sample_model, batch, noise_t)
        denoised.append(result.cpu())
    return torch.cat(denoised, dim=0)


def main():
    # Extract custom args before hparams parser sees them
    custom_args = {}
    argv = sys.argv[:]
    for flag in ['--result_dir', '--num_images', '--eval_snrs', '--jscc_dir']:
        if flag in argv:
            idx = argv.index(flag)
            custom_args[flag.replace('--', '')] = argv[idx + 1]
            del argv[idx:idx + 2]
    sys.argv = argv

    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up memory-efficient sampling for {H.sampler}')
    start_training_log(H)

    result_dir = custom_args.get('result_dir', H.log_dir)
    num_images = int(custom_args.get('num_images', 100))
    proc_batch = 8  # batch size for BAE encode/decode — small enough to avoid OOM
    os.makedirs(result_dir, exist_ok=True)

    misc.init_distributed_mode(H)

    Eb_N0_db_range = torch.tensor([H.snr_range[0], H.snr_range[1]])
    Eb_N0_dB_value = torch.linspace(Eb_N0_db_range[1], Eb_N0_db_range[0], 64)

    # Parse evaluation SNR points (comma-separated). Default: all 64 steps
    eval_snrs_str = custom_args.get('eval_snrs', None)
    if eval_snrs_str is not None:
        eval_snrs = [float(x) for x in eval_snrs_str.split(',')]
        # Map each eval SNR to the closest noise_t index
        eval_points = []
        for snr in eval_snrs:
            diff = torch.abs(Eb_N0_dB_value - snr)
            noise_t = int(diff.argmin().item())
            actual_snr = Eb_N0_dB_value[noise_t].item()
            eval_points.append((noise_t, actual_snr))
    else:
        eval_points = [(t, Eb_N0_dB_value[t].item()) for t in range(63, -1, -1)]

    # LDM config
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
    config = OmegaConf.load(base_configs[0])

    device = torch.device('cuda')

    # Load BAE
    print("Loading BAE...")
    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H, ['encoder', 'quantize', 'generator'], remove_component_from_key=False
    )
    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.to(device)
    bergan.eval()
    del ae_state_dict

    # Load sampler
    print("Loading sampler...")
    sampler = get_sampler(H, None).to(device)
    if H.ema:
        ema_sampler = copy.deepcopy(sampler)
    if H.distributed:
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=False)
        sampler_without_ddp = sampler.module

    if H.load_step > 0:
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=True).to(device)
        if H.ema:
            try:
                ema_sampler = load_model(ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=True)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)

    sample_model = ema_sampler if H.ema else (sampler_without_ddp if H.distributed else sampler)
    sample_model.eval()

    # Dataset
    val_dataset = instantiate_from_config(config.data.params.validation)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=2, sampler=torch.utils.data.SequentialSampler(val_dataset),
        batch_size=proc_batch, pin_memory=True, drop_last=False,
    )

    # Collect num_images validation images
    print(f"Collecting {num_images} validation images...")
    val_images = []
    count = 0
    for val_batch in val_loader:
        if count >= num_images:
            break
        take = min(val_batch["image"].size(0), num_images - count)
        img = val_batch["image"][:take]
        img = img / 255.0
        img = img.permute(0, 3, 1, 2)
        val_images.append(img)
        count += take
    val_img = torch.cat(val_images, dim=0)

    # Encode to binary codes (batched)
    print(f"Encoding {num_images} images to binary codes (batch={proc_batch})...")
    code_val = batched_encode(bergan, val_img, device, batch_size=proc_batch)
    b, c, h, w = code_val.shape
    x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()

    qam_str = f"{H.qam_order}QAM"

    # ---- 1. Save GT ----
    print("Saving GT images...")
    gt_dir = f"{result_dir}/samples_for_psnr/{qam_str}/gt_{H.load_step}"
    os.makedirs(gt_dir, exist_ok=True)
    gt_images = batched_decode(bergan, x_val, H, device, batch_size=proc_batch)
    for idx, img in enumerate(gt_images):
        torchvision.utils.save_image(torch.clamp(img, 0, 1), f"{gt_dir}/{idx}.png")

    # ---- 2. For each SNR: save noisy AND denoised images ----
    print(f"Processing {len(eval_points)} SNR levels (batch={proc_batch})...")
    for idx, (noise_t, eb_n0_db) in enumerate(eval_points):
        db_str = f"{eb_n0_db:.2f}_db"

        noise_dir = f"{result_dir}/samples_for_psnr/{qam_str}/noise/{db_str}"
        denoise_dir = f"{result_dir}/samples_for_psnr/{qam_str}/denoise/{db_str}"
        os.makedirs(noise_dir, exist_ok=True)
        os.makedirs(denoise_dir, exist_ok=True)

        print(f"  SNR {eb_n0_db:.2f} dB ({idx+1}/{len(eval_points)})")

        img_idx = 0
        for start in range(0, num_images, proc_batch):
            end = min(start + proc_batch, num_images)
            x_batch = x_val[start:end].to(device)

            # Add QAM noise
            x_noisy = make_code_noise(x_batch, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=device)

            # Noisy → decode and save
            noisy_imgs = get_online_decode_code_into_images(x_noisy, bergan, H)
            for img in noisy_imgs:
                torchvision.utils.save_image(torch.clamp(img, 0, 1), f"{noise_dir}/{img_idx}.png")
                img_idx += 1

        # Denoise all codes at once (latent codes are small, fits in GPU)
        x_val_gpu = x_val.to(device)
        x_noisy_all = make_code_noise(x_val_gpu, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=device)
        x_denoised_all = batched_denoise(sample_model, x_noisy_all, noise_t, device, batch_size=proc_batch)
        denoised_imgs = batched_decode(bergan, x_denoised_all, H, device, batch_size=proc_batch)
        for idx, img in enumerate(denoised_imgs):
            torchvision.utils.save_image(torch.clamp(img, 0, 1), f"{denoise_dir}/{idx}.png")

        # Free GPU memory between SNR levels
        torch.cuda.empty_cache()

    print(f"\nImages saved to {result_dir}/samples_for_psnr/{qam_str}/")

    # =========================================================================
    # Step 3: Compute PSNR & SSIM metrics
    # =========================================================================
    print("\nComputing metrics...")
    sample_dir = f"{result_dir}/samples_for_psnr/{qam_str}"
    metrics_dir = f"{result_dir}/metrics_results/{qam_str}"

    for mode in ["denoise", "noise"]:
        mode_path = os.path.join(sample_dir, mode)
        if not os.path.exists(mode_path):
            print(f"Warning: {mode} directory not found at {mode_path}")
            continue

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
                gt_path = os.path.join(gt_dir, img_name)

                if not os.path.exists(gt_path):
                    continue

                db_img = cv2.imread(db_img_path)
                gt_img = cv2.imread(gt_path)

                if db_img is None or gt_img is None:
                    continue

                try:
                    p_val = psnr(gt_img, db_img, data_range=255)
                    s_val = ssim(gt_img, db_img, channel_axis=2, win_size=11, data_range=255)
                    psnr_vals.append(p_val)
                    ssim_vals.append(s_val)
                except Exception:
                    continue

            if psnr_vals:
                snr_key = float(db_folder.split('_')[0])
                psnr_results[snr_key] = (np.mean(psnr_vals), np.std(psnr_vals))
                ssim_results[snr_key] = (np.mean(ssim_vals), np.std(ssim_vals))
                print(f"  [{mode}] SNR={snr_key:.2f} dB: PSNR={psnr_results[snr_key][0]:.4f}, SSIM={ssim_results[snr_key][0]:.4f}")

        for metric_name, results in [("psnr", psnr_results), ("ssim", ssim_results)]:
            save_path = os.path.join(metrics_dir, mode, metric_name)
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f"{metric_name}.txt")
            with open(filepath, "w") as f:
                for snr in sorted(results.keys()):
                    mean_val, std_val = results[snr]
                    f.write(f"{snr:.2f}_db: {mean_val:.4f} +/- {std_val:.4f}\n")
            print(f"  Saved {filepath}")

    # =========================================================================
    # Step 4: Plot (including DeepJSCC baseline comparison)
    # =========================================================================
    print("\nGenerating plots...")

    jscc_dir = custom_args.get('jscc_dir', '')
    jscc_train_snrs = [-15, 0, 4, 8, 25]

    def parse_jscc_txt(file_path):
        """Parse JSCC psnr_summary.txt files."""
        x_data, y_data = [], []
        if not os.path.exists(file_path):
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

    def read_metrics(file_path):
        snr_vals, metric_vals = [], []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(': ')
                if len(parts) >= 2:
                    snr = float(parts[0].replace('_db', ''))
                    metric = float(parts[1].split(' +/- ')[0])
                    snr_vals.append(snr)
                    metric_vals.append(metric)
        idx = np.argsort(snr_vals)
        return np.array(snr_vals)[idx], np.array(metric_vals)[idx]

    pdf_dir = f"{result_dir}/pdf"
    os.makedirs(pdf_dir, exist_ok=True)

    denoise_psnr_snr, denoise_psnr_vals = read_metrics(os.path.join(metrics_dir, "denoise", "psnr", "psnr.txt"))
    denoise_ssim_snr, denoise_ssim_vals = read_metrics(os.path.join(metrics_dir, "denoise", "ssim", "ssim.txt"))
    noise_psnr_snr, noise_psnr_vals = read_metrics(os.path.join(metrics_dir, "noise", "psnr", "psnr.txt"))
    noise_ssim_snr, noise_ssim_vals = read_metrics(os.path.join(metrics_dir, "noise", "ssim", "ssim.txt"))

    label = "BLD_DSC (AdaLN)"

    # --- Academic-style plot: PSNR comparison with JSCC ---
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    jscc_colors = ['#9467bd', '#17becf', '#2ca02c', '#ff7f0e', '#7f7f7f']
    jscc_markers = ['v', '^', '<', '>', 's']

    # Plot JSCC baselines
    if jscc_dir and os.path.exists(jscc_dir):
        for idx, train_snr in enumerate(jscc_train_snrs):
            path = os.path.join(jscc_dir, f"fix_snr_{train_snr}", "psnr_summary.txt")
            x, y = parse_jscc_txt(path)
            if len(x) > 0:
                ax.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                        markersize=8, alpha=0.6, color=jscc_colors[idx],
                        label=f'JSCC: {train_snr}dB')
    else:
        print(f"  Note: JSCC dir not found ({jscc_dir}), skipping JSCC overlay.")

    # Plot Ours (denoised)
    ours_x, ours_y = denoise_psnr_snr, denoise_psnr_vals
    print(f"  Ours denoised PSNR: SNR {ours_x} -> {ours_y}")
    ax.plot(ours_x, ours_y, linestyle='-', linewidth=4, marker='o',
            markersize=10, color='#d62728', markeredgecolor='black',
            label=label, zorder=10)

    ax.set_title('PSNR Performance across Channel SNR Range')
    ax.set_xlabel('Test Channel SNR (dB)')
    ax.set_ylabel('Average PSNR (dB)')
    ax.set_xlim([-17, 27])
    ax.set_xticks(np.arange(-15, 25, 5))
    ax.set_ylim([5, 35])
    ax.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, "JSCC_vs_Ours_psnr.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # --- SSIM comparison with JSCC ---
    fig2, ax2 = plt.subplots(figsize=(11, 8), dpi=300)

    if jscc_dir and os.path.exists(jscc_dir):
        for idx, train_snr in enumerate(jscc_train_snrs):
            path = os.path.join(jscc_dir, f"fix_snr_{train_snr}", "ssim_summary.txt")
            x, y = parse_jscc_txt(path)
            if len(x) > 0:
                ax2.plot(x, y, linestyle='--', linewidth=1.5, marker=jscc_markers[idx],
                         markersize=8, alpha=0.6, color=jscc_colors[idx],
                         label=f'JSCC: {train_snr}dB')

    ax2.plot(denoise_ssim_snr, denoise_ssim_vals, linestyle='-', linewidth=4, marker='o',
             markersize=10, color='#d62728', markeredgecolor='black',
             label=label, zorder=10)
    ax2.set_title('SSIM Performance across Channel SNR Range')
    ax2.set_xlabel('Test Channel SNR (dB)')
    ax2.set_ylabel('Average SSIM')
    ax2.set_xlim([-17, 27])
    ax2.set_xticks(np.arange(-15, 25, 5))
    ax2.legend(loc='upper left', frameon=True, shadow=True, edgecolor='black')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, "JSCC_vs_Ours_ssim.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # --- Simple denoised-only PSNR + SSIM (no JSCC) ---
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
                         'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

    plt.figure(figsize=(8, 5))
    plt.plot(denoise_psnr_snr, denoise_psnr_vals, 'b-o', linewidth=1.5, markersize=5, label=f'{label} (denoised)')
    plt.plot(noise_psnr_snr, noise_psnr_vals, 'r-s', linewidth=1.5, markersize=5, label=f'{label} (noisy)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR vs SNR — {label}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, "psnr_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(denoise_ssim_snr, denoise_ssim_vals, 'b-o', linewidth=1.5, markersize=5, label=f'{label} (denoised)')
    plt.plot(noise_ssim_snr, noise_ssim_vals, 'r-s', linewidth=1.5, markersize=5, label=f'{label} (noisy)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('SSIM')
    plt.title(f'SSIM vs SNR — {label}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, "ssim_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 8))
    ax3.plot(denoise_psnr_snr, denoise_psnr_vals, 'b-o', linewidth=1.5, markersize=5)
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('PSNR (dB)')
    ax3.set_title(f'{label} — Denoised PSNR')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax4.plot(denoise_ssim_snr, denoise_ssim_vals, 'r-s', linewidth=1.5, markersize=5)
    ax4.set_xlabel('SNR (dB)')
    ax4.set_ylabel('SSIM')
    ax4.set_title(f'{label} — Denoised SSIM')
    ax4.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, "denoise_metrics.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {pdf_dir}/")


if __name__ == '__main__':
    main()
