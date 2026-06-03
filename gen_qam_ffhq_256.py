import torch
import numpy as np
import copy
import os
from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
    get_sampler, get_online_samples_denoise_code, get_online_decode_code_into_images
from utils.log_utils import log, config_log, start_training_log, load_model, save_images
import misc
import torch.distributed as dist
from omegaconf import OmegaConf
import glob
from ldm.ldm.util import instantiate_from_config
from utils.qam_utils import *
from utils.m_qam_awgn_util import *
import torchvision


def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
    received_code = make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
    return received_code


def main(H, vis):

    misc.init_distributed_mode(H)

    Eb_N0_db_range = torch.tensor([H.snr_range[0], H.snr_range[1]])
    Eb_N0_dB_value = torch.linspace(Eb_N0_db_range[1], Eb_N0_db_range[0], 64)

    # LDM ae config
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = configs[0]

    # Load BAE
    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )
    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict

    # Load sampler
    sampler = get_sampler(H, None).cuda()
    if H.ema:
        ema_sampler = copy.deepcopy(sampler)
    if H.distributed:
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=False)
        sampler_without_ddp = sampler.module

    if H.load_step > 0:
        device = sampler.device if hasattr(sampler, 'device') else 'cuda'
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=True).cuda()
        if H.ema:
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=True)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)

    # Dataset
    val_dataset = instantiate_from_config(config.data.params.validation)
    if H.distributed:
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=4, sampler=sampler_val,
        batch_size=H.batch_size, pin_memory=True, drop_last=False,
    )

    # Collect num_images validation images
    num_images = 100
    val_images = []
    count = 0
    with torch.no_grad():
        for val_batch in val_loader:
            if count >= num_images:
                break
            batch_size = val_batch["image"].size(0)
            take = min(batch_size, num_images - count)
            val_img_batch = val_batch["image"][:take].cuda()
            val_img_batch = val_img_batch / 255.0
            val_img_batch = val_img_batch.permute(0, 3, 1, 2)
            val_images.append(val_img_batch)
            count += take

    val_img = torch.cat(val_images, dim=0)

    # Encode all images to binary codes
    with torch.no_grad():
        code_val = bergan(val_img, code_only=True).detach()
        b, c, h, w = code_val.shape
        x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()

    qam_str = f"{H.qam_order}QAM"
    sample_model = ema_sampler if H.ema else (sampler_without_ddp if H.distributed else sampler)

    # ---- 1. Save GT (clean code → decode, no channel) ----
    gt_dir = f"{H.log_dir}/samples_for_psnr/{qam_str}/gt_{H.load_step}"
    os.makedirs(gt_dir, exist_ok=True)
    gt_images = get_online_decode_code_into_images(x_val, bergan, H)
    for idx in range(len(gt_images)):
        torchvision.utils.save_image(torch.clamp(gt_images[idx], 0, 1), f"{gt_dir}/{idx}.png")

    # ---- 2. For each SNR: save noisy AND denoised images ----
    for noise_t in range(63, -1, -1):
        eb_n0_db = Eb_N0_dB_value[noise_t]
        db_str = f"{eb_n0_db:.2f}_db"

        noise_dir = f"{H.log_dir}/samples_for_psnr/{qam_str}/noise/{db_str}"
        denoise_dir = f"{H.log_dir}/samples_for_psnr/{qam_str}/denoise/{db_str}"
        os.makedirs(noise_dir, exist_ok=True)
        os.makedirs(denoise_dir, exist_ok=True)

        print(f"Processing SNR {eb_n0_db:.2f} dB  (step {noise_t})")

        global_idx = 0
        for start_idx in range(0, num_images, H.batch_size):
            end_idx = min(start_idx + H.batch_size, num_images)
            x_batch = x_val[start_idx:end_idx]

            # Add QAM noise
            x_noisy = make_code_noise(x_batch, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=x_batch.device)

            # Noisy → decode
            noisy_imgs = get_online_decode_code_into_images(x_noisy, bergan, H)
            for idx in range(len(noisy_imgs)):
                torchvision.utils.save_image(torch.clamp(noisy_imgs[idx], 0, 1), f"{noise_dir}/{global_idx}.png")
                global_idx += 1

            # Noisy → diffusion denoise → decode
            x_denoised = get_online_samples_denoise_code(sample_model, x_noisy, noise_t)
            denoised_imgs = get_online_decode_code_into_images(x_denoised, bergan, H)
            for idx in range(len(denoised_imgs)):
                torchvision.utils.save_image(torch.clamp(denoised_imgs[idx], 0, 1), f"{denoise_dir}/{global_idx - len(noisy_imgs) + idx}.png")

    print(f"Done. Images saved to {H.log_dir}/samples_for_psnr/{qam_str}/")


if __name__ == '__main__':
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up sampling for {H.sampler}')
    start_training_log(H)
    main(H, None)
