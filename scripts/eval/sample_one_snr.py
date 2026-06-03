"""Quickly sample one SNR point for AdaLN ablation. Usage: python scripts/eval/sample_one_snr.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch, numpy as np, copy, glob
import torchvision
from models.binaryae import BinaryAutoEncoder
from hparams import get_sampler_hparams
from utils.sampler_utils import (retrieve_autoencoder_components_state_dicts, get_sampler,
                                  get_online_samples_denoise_code, get_online_decode_code_into_images)
from utils.log_utils import load_model
from omegaconf import OmegaConf
from ldm.ldm.util import instantiate_from_config
from utils.qam_utils import *
from utils.m_qam_awgn_util import *
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Config
RESULT_DIR = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln"
WEIGHT_DIR = RESULT_DIR
AE_DIR = "/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64"
AE_STEP = 8100000
LOAD_STEP = 50000
TARGET_SNR = 15.0
DEVICE = torch.device('cuda')
PROC_BATCH = 8

# Override sys.argv for hparams
sys.argv = [sys.argv[0],
    '--sampler', 'bld_dsc', '--dataset', 'ffhq', '--ema',
    '--codebook_size', '64', '--img_size', '256', '--total_steps', '64', '--sample_steps', '64',
    '--beta_type', 'linear', '--amp', '--latent_shape', '1', '16', '16',
    '--loss_final', 'mean', '--p_flip', '--norm_first', '--qam_order', '16',
    '--snr_range', '0', '15', '--time_cond_mode', 'adaln',
    '--ae_load_dir', AE_DIR, '--ae_load_step', str(AE_STEP),
    '--batch_size', '16', '--log_dir', RESULT_DIR,
    '--load_step', str(LOAD_STEP), '--load_dir', WEIGHT_DIR,
]
H = get_sampler_hparams()

# Find noise_t for target SNR
Eb_N0_dB_value = torch.linspace(H.snr_range[1], H.snr_range[0], 64)
noise_t = int(torch.abs(Eb_N0_dB_value - TARGET_SNR).argmin().item())
actual_snr = Eb_N0_dB_value[noise_t].item()
print(f"Target SNR={TARGET_SNR} -> noise_t={noise_t}, actual SNR={actual_snr:.4f}")

# Load BAE
ldm_ae_dir = "./ldm/models/ldm/ffhq256"
config = OmegaConf.load(sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))[0])
ae_sd = retrieve_autoencoder_components_state_dicts(H, ['encoder', 'quantize', 'generator'], remove_component_from_key=False)
bergan = BinaryAutoEncoder(H); bergan.load_state_dict(ae_sd, strict=True); bergan = bergan.to(DEVICE).eval(); del ae_sd

# Load sampler (no DDP, strip module. prefix manually)
def load_without_ddp(model, model_name, step, load_dir, device):
    path = os.path.join(load_dir, 'saved_models', f'{model_name}_{step}.th')
    print(f"Loading {path}")
    sd = torch.load(path, map_location=device)
    # Strip 'module.' prefix from DDP-wrapped checkpoint
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    del sd, new_sd
    return model

sampler_raw = get_sampler(H, None)
sampler = load_without_ddp(sampler_raw, H.sampler, LOAD_STEP, WEIGHT_DIR, DEVICE).to(DEVICE)
ema_sampler = copy.deepcopy(sampler_raw)
ema_sampler = load_without_ddp(ema_sampler, f'{H.sampler}_ema', LOAD_STEP, WEIGHT_DIR, DEVICE).to(DEVICE)
ema_sampler.eval()

# Get val images
val_dataset = instantiate_from_config(config.data.params.validation)
val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=2, sampler=torch.utils.data.SequentialSampler(val_dataset), batch_size=PROC_BATCH, pin_memory=True, drop_last=False)
val_imgs, count = [], 0
for batch in val_loader:
    if count >= 100: break
    take = min(batch["image"].size(0), 100 - count)
    img = batch["image"][:take] / 255.0; img = img.permute(0, 3, 1, 2)
    val_imgs.append(img); count += take
val_img = torch.cat(val_imgs, dim=0)

# Encode
codes = []
for i in range(0, val_img.shape[0], PROC_BATCH):
    b = val_img[i:i+PROC_BATCH].to(DEVICE)
    with torch.no_grad(): codes.append(bergan(b, code_only=True).detach().cpu())
codes = torch.cat(codes, dim=0)
b, c, h, w = codes.shape; x_val = codes.view(b, c, -1).permute(0, 2, 1).contiguous()

# Sample at target SNR
qam_str = f"{H.qam_order}QAM"
db_str = f"{actual_snr:.2f}_db"
noise_dir = f"{RESULT_DIR}/samples_for_psnr/{qam_str}/noise/{db_str}"
denoise_dir = f"{RESULT_DIR}/samples_for_psnr/{qam_str}/denoise/{db_str}"
os.makedirs(noise_dir, exist_ok=True); os.makedirs(denoise_dir, exist_ok=True)

# GT dir
gt_dir = f"{RESULT_DIR}/samples_for_psnr/{qam_str}/gt_{LOAD_STEP}"

# Noisy images
for start in range(0, 100, PROC_BATCH):
    end = min(start + PROC_BATCH, 100)
    xb = x_val[start:end].to(DEVICE)
    x_noisy = make_code_noise_single(xb, eb_n0_db=actual_snr, qam_order=H.qam_order, device=DEVICE)
    noisy_imgs = get_online_decode_code_into_images(x_noisy, bergan, H)
    for j, img in enumerate(noisy_imgs):
        torchvision.utils.save_image(torch.clamp(img, 0, 1), f"{noise_dir}/{start+j}.png")

# Denoised images
x_val_gpu = x_val.to(DEVICE)
x_noisy_all = make_code_noise_single(x_val_gpu, eb_n0_db=actual_snr, qam_order=H.qam_order, device=DEVICE)
denoised_codes = []
for i in range(0, 100, PROC_BATCH):
    with torch.no_grad(): denoised_codes.append(get_online_samples_denoise_code(ema_sampler, x_noisy_all[i:i+PROC_BATCH], noise_t).cpu())
x_denoised = torch.cat(denoised_codes, dim=0)
denoised_imgs = []
for i in range(0, 100, PROC_BATCH):
    with torch.no_grad():
        imgs = get_online_decode_code_into_images(x_denoised[i:i+PROC_BATCH].to(DEVICE), bergan, H)
    denoised_imgs.extend([img.cpu() for img in imgs])
for j, img in enumerate(denoised_imgs):
    torchvision.utils.save_image(torch.clamp(img, 0, 1), f"{denoise_dir}/{j}.png")

# Compute metrics
psnr_vals, ssim_vals = [], []
for j in range(100):
    db_img = cv2.imread(f"{denoise_dir}/{j}.png")
    gt_img = cv2.imread(f"{gt_dir}/{j}.png")
    if db_img is not None and gt_img is not None:
        psnr_vals.append(psnr(gt_img, db_img, data_range=255))
        ssim_vals.append(ssim(gt_img, db_img, channel_axis=2, win_size=11, data_range=255))

print(f"SNR {actual_snr:.4f} dB: PSNR={np.mean(psnr_vals):.4f} +/- {np.std(psnr_vals):.4f}, SSIM={np.mean(ssim_vals):.4f} +/- {np.std(ssim_vals):.4f}")

# Append to existing metrics files
def append_metric(filepath, snr, mean_val, std_val):
    lines = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f: lines = f.readlines()
    # Remove existing entry for this SNR if present
    prefix = f"{snr:.2f}_db:"
    lines = [l for l in lines if not l.startswith(prefix)]
    lines.append(f"{snr:.2f}_db: {mean_val:.4f} +/- {std_val:.4f}\n")
    lines.sort(key=lambda x: float(x.split('_db')[0]))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f: f.writelines(lines)

append_metric(f"{RESULT_DIR}/metrics_results/16QAM/denoise/psnr/psnr.txt", actual_snr, np.mean(psnr_vals), np.std(psnr_vals))
append_metric(f"{RESULT_DIR}/metrics_results/16QAM/denoise/ssim/ssim.txt", actual_snr, np.mean(ssim_vals), np.std(ssim_vals))
print(f"Metrics updated. Now re-run plot: python scripts/eval/plot_jscc_comparison.py")
