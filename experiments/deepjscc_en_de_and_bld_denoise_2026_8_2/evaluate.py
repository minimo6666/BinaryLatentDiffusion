#!/usr/bin/env python3
"""Paired evaluation of frozen JSCC-25 with and without BLD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, SequentialSampler

from bld_model import ExactQAMAWGNBinaryDiffusionDSC
from common import (
    DEFAULT_JSCC_CHECKPOINT,
    DEFAULT_REFERENCE_PSNR,
    DEFAULT_RESULTS_DIR,
    build_hparams,
    decode_sequence,
    encode_clean_sequence,
    image_batch_to_float,
    load_ffhq_dataset,
    load_frozen_jscc,
    parse_snrs,
    save_json,
    seed_everything,
)
from models.transformer import TransformerBD
from utils.m_qam_awgn_util import make_code_noise_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RESULTS_DIR / "checkpoints/latest.pt")
    parser.add_argument("--jscc-checkpoint", type=Path, default=DEFAULT_JSCC_CHECKPOINT)
    parser.add_argument("--reference-psnr", type=Path, default=DEFAULT_REFERENCE_PSNR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--eval-snrs", default="0,4,8,12,15,25")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--save-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--denoise-mode", choices=["iterative", "direct"], default="iterative")
    return parser.parse_args()


def psnr_per_image(reference: torch.Tensor, estimate: torch.Tensor) -> np.ndarray:
    mse = (reference - estimate).square().flatten(1).mean(1).clamp_min(1e-12)
    return (-10.0 * torch.log10(mse)).detach().cpu().numpy()


def ssim_per_image(reference: torch.Tensor, estimate: torch.Tensor) -> np.ndarray:
    ref = reference.detach().cpu().permute(0, 2, 3, 1).numpy()
    est = estimate.detach().cpu().permute(0, 2, 3, 1).numpy()
    return np.asarray([
        structural_similarity(r, e, data_range=1.0, channel_axis=2)
        for r, e in zip(ref, est)
    ])


def timestep_for_snr(sampler, snr: float) -> tuple[int, float, bool]:
    schedule = sampler.Eb_N0_dB_values.detach().cpu()
    clamped = min(max(snr, float(schedule.min())), float(schedule.max()))
    timestep = int(torch.abs(schedule - clamped).argmin().item())
    return timestep, float(schedule[timestep].item()), not math.isclose(snr, clamped)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    seed_everything(args.seed)
    device = torch.device("cuda")
    result_dir = args.result_dir.resolve()
    metrics_dir = result_dir / "metrics"
    samples_dir = result_dir / "samples"
    reference_dir = result_dir / "reference"
    for directory in (metrics_dir, samples_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(state["config"])
    # The checkpoint config also contains run metadata not consumed by hparams.
    hparams = build_hparams(model_config)
    autoencoder = load_frozen_jscc(args.jscc_checkpoint, hparams, device)
    denoiser = TransformerBD(hparams).to(device)
    sampler = ExactQAMAWGNBinaryDiffusionDSC(
        hparams, denoiser, hparams.codebook_size
    ).to(device)
    sampler.load_state_dict(state["ema_sampler"], strict=True)
    sampler.eval()
    del state

    if args.reference_psnr.is_file():
        shutil.copy2(args.reference_psnr, reference_dir / "original_jscc25_psnr_summary.txt")

    dataset = load_ffhq_dataset("validation")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    images_cpu = []
    for data in loader:
        remaining = args.num_images - sum(batch.shape[0] for batch in images_cpu)
        if remaining <= 0:
            break
        images_cpu.append(image_batch_to_float(data)[:remaining].cpu())
    images_cpu = torch.cat(images_cpu, dim=0)
    if images_cpu.shape[0] != args.num_images:
        raise RuntimeError(f"Requested {args.num_images} images, got {images_cpu.shape[0]}")

    clean_codes = []
    clean_psnr = []
    for start in range(0, args.num_images, args.batch_size):
        images = images_cpu[start:start + args.batch_size].to(device)
        x0 = encode_clean_sequence(autoencoder, images)
        clean_codes.append(x0.cpu())
        clean_reconstruction = decode_sequence(autoencoder, x0, hparams).clamp(0.0, 1.0)
        clean_psnr.extend(psnr_per_image(images, clean_reconstruction).tolist())
    clean_codes = torch.cat(clean_codes, dim=0)

    rows = []
    eval_snrs = parse_snrs(args.eval_snrs)
    for snr_index, snr in enumerate(eval_snrs):
        # One deterministic channel draw is shared by baseline and BLD.
        torch.manual_seed(args.seed + 1000 * snr_index)
        baseline_psnr, bld_psnr = [], []
        baseline_ssim, bld_ssim = [], []
        noisy_errors = 0
        bld_errors = 0
        bit_count = 0
        timestep, schedule_snr, out_of_range = timestep_for_snr(sampler, snr)
        snr_sample_dir = samples_dir / f"{snr:g}_db"
        (snr_sample_dir / "gt").mkdir(parents=True, exist_ok=True)
        (snr_sample_dir / "jscc25").mkdir(parents=True, exist_ok=True)
        (snr_sample_dir / "jscc25_bld").mkdir(parents=True, exist_ok=True)

        for start in range(0, args.num_images, args.batch_size):
            end = min(start + args.batch_size, args.num_images)
            images = images_cpu[start:end].to(device)
            x0 = clean_codes[start:end].to(device)
            x_noisy = make_code_noise_single(
                x0,
                eb_n0_db=float(snr),
                qam_order=hparams.qam_order,
                device=device,
            )
            baseline_images = decode_sequence(autoencoder, x_noisy, hparams).clamp(0.0, 1.0)

            if args.denoise_mode == "direct":
                x_bld = sampler.predict_x0(x_noisy, timestep)
            else:
                # timestep=0 intentionally bypasses BLD at/above the clean end of the schedule.
                x_bld = sampler.sample(x_noisy, timestep)
            bld_images = decode_sequence(autoencoder, x_bld, hparams).clamp(0.0, 1.0)

            baseline_psnr.extend(psnr_per_image(images, baseline_images).tolist())
            bld_psnr.extend(psnr_per_image(images, bld_images).tolist())
            baseline_ssim.extend(ssim_per_image(images, baseline_images).tolist())
            bld_ssim.extend(ssim_per_image(images, bld_images).tolist())
            noisy_errors += int(((x_noisy > 0.5) != (x0 > 0.5)).sum().item())
            bld_errors += int(((x_bld > 0.5) != (x0 > 0.5)).sum().item())
            bit_count += x0.numel()

            for local_index in range(end - start):
                global_index = start + local_index
                if global_index >= args.save_images:
                    continue
                torchvision.utils.save_image(images[local_index], snr_sample_dir / "gt" / f"{global_index:04d}.png")
                torchvision.utils.save_image(baseline_images[local_index], snr_sample_dir / "jscc25" / f"{global_index:04d}.png")
                torchvision.utils.save_image(bld_images[local_index], snr_sample_dir / "jscc25_bld" / f"{global_index:04d}.png")

        row = {
            "test_ebn0_db": snr,
            "bld_schedule_ebn0_db": schedule_snr,
            "outside_training_range": int(out_of_range),
            "jscc25_psnr_mean": float(np.mean(baseline_psnr)),
            "jscc25_psnr_std": float(np.std(baseline_psnr)),
            "jscc25_bld_psnr_mean": float(np.mean(bld_psnr)),
            "jscc25_bld_psnr_std": float(np.std(bld_psnr)),
            "psnr_gain_db": float(np.mean(bld_psnr) - np.mean(baseline_psnr)),
            "jscc25_ssim_mean": float(np.mean(baseline_ssim)),
            "jscc25_ssim_std": float(np.std(baseline_ssim)),
            "jscc25_bld_ssim_mean": float(np.mean(bld_ssim)),
            "jscc25_bld_ssim_std": float(np.std(bld_ssim)),
            "jscc25_ber": noisy_errors / bit_count,
            "jscc25_bld_ber": bld_errors / bit_count,
        }
        rows.append(row)
        print(
            f"Eb/N0={snr:6.2f} dB | JSCC-25={row['jscc25_psnr_mean']:.4f} dB | "
            f"JSCC-25+BLD={row['jscc25_bld_psnr_mean']:.4f} dB | "
            f"gain={row['psnr_gain_db']:+.4f} dB"
        )

    csv_path = metrics_dir / "paired_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        metrics_dir / "evaluation_metadata.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "jscc_checkpoint": str(args.jscc_checkpoint.resolve()),
            "num_images": args.num_images,
            "saved_images_per_snr": args.save_images,
            "denoise_mode": args.denoise_mode,
            "qam_order": hparams.qam_order,
            "channel_axis": "Eb/N0 (dB)",
            "clean_jscc25_psnr_mean": float(np.mean(clean_psnr)),
            "clean_jscc25_psnr_std": float(np.std(clean_psnr)),
            "paired_noise": True,
            "note": "SNRs outside [0, 15] use the closest BLD schedule endpoint.",
        },
    )
    print(f"Saved paired metrics: {csv_path}")


if __name__ == "__main__":
    main()
