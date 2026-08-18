#!/usr/bin/env python3
"""Evaluate the physical-QAM BFM on the shared 100-image FFHQ manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_MANIFEST,
    EXPERIMENT_DIR,
    bit_error_rate,
    build_hparams,
    code_to_sequence,
    create_sampler,
    decode_sequence,
    image_batch,
    load_autoencoder,
    load_bfm_checkpoint,
    load_test_manifest,
    make_dataset,
    per_image_psnr,
    safe_snr_name,
    save_bfm_preview,
    save_tensor_image,
    seed_everything,
    transmit_bits,
)
from utils.m_qam_awgn_util import ebn0_db_to_noise_sigma
from utils.qam_utils import general_m_qam_ber

DEFAULT_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EXPERIMENT_DIR / "checkpoints" / "bfm_qam_ema_final.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--qam-order", type=int, default=16)
    parser.add_argument("--snr-min", type=float, default=-15)
    parser.add_argument("--snr-max", type=float, default=20)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-images", type=int, default=100)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    seed_everything(args.seed)

    manifest = load_test_manifest(args.manifest)
    dataset = make_dataset("validation", names=manifest["filenames"])
    if len(dataset) != 100:
        raise RuntimeError(f"Expected exactly 100 held-out images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    hparams = build_hparams(
        qam_order=args.qam_order,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
    )
    sampler = create_sampler(hparams, device)
    checkpoint_payload = load_bfm_checkpoint(sampler, args.checkpoint)
    sampler.eval()
    autoencoder = load_autoencoder(args.ae_checkpoint, device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accumulators = {
        float(snr): {
            "clean_psnr": [],
            "baseline_psnr": [],
            "direct_psnr": [],
            "denoised_psnr": [],
            "channel_errors": 0,
            "direct_errors": 0,
            "denoised_errors": 0,
            "bit_count": 0,
        }
        for snr in args.snrs
    }
    per_image_rows = []
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = image_batch(batch, device)
            names = list(batch["name"])
            clean_bits = code_to_sequence(
                autoencoder(images, code_only=True).detach()
            )
            clean_images = decode_sequence(autoencoder, clean_bits).clamp(0, 1)
            clean_psnr = per_image_psnr(images, clean_images)

            for snr_index, snr_value in enumerate(args.snrs):
                snr = float(snr_value)
                channel_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_003 * snr_index + batch_index
                )
                received = transmit_bits(
                    clean_bits, snr, args.qam_order, generator=channel_generator
                )
                path_start = sampler.path_step_from_snr(snr)
                direct = sampler.predict_clean_bits_direct(
                    received, path_start, temperature=args.temperature
                )
                torch.manual_seed(args.seed + 10_000_019 * snr_index + batch_index)
                torch.cuda.manual_seed_all(
                    args.seed + 10_000_019 * snr_index + batch_index
                )
                denoised = sampler.sample_from_channel(
                    received,
                    path_start,
                    temp=args.temperature,
                    sample_steps=max(1, path_start),
                )
                baseline_images = decode_sequence(autoencoder, received).clamp(0, 1)
                direct_images = decode_sequence(autoencoder, direct).clamp(0, 1)
                denoised_images = decode_sequence(autoencoder, denoised).clamp(0, 1)
                raw_ber = bit_error_rate(clean_bits, received)
                direct_ber = bit_error_rate(clean_bits, direct)
                final_ber = bit_error_rate(clean_bits, denoised)
                baseline_psnr = per_image_psnr(images, baseline_images)
                direct_psnr = per_image_psnr(images, direct_images)
                denoised_psnr = per_image_psnr(images, denoised_images)

                metrics = accumulators[snr]
                metrics["clean_psnr"].extend(clean_psnr.cpu().tolist())
                metrics["baseline_psnr"].extend(baseline_psnr.cpu().tolist())
                metrics["direct_psnr"].extend(direct_psnr.cpu().tolist())
                metrics["denoised_psnr"].extend(denoised_psnr.cpu().tolist())
                metrics["channel_errors"] += int((clean_bits != received).sum())
                metrics["direct_errors"] += int((clean_bits != direct).sum())
                metrics["denoised_errors"] += int((clean_bits != denoised).sum())
                metrics["bit_count"] += clean_bits.numel()

                preview_rows = []
                for local_index, filename in enumerate(names):
                    raw = float(raw_ber[local_index])
                    direct_value = float(direct_ber[local_index])
                    final = float(final_ber[local_index])
                    row = {
                        "filename": filename,
                        "eval_ebn0_db": snr,
                        "path_start": path_start,
                        "clean_psnr_db": float(clean_psnr[local_index]),
                        "baseline_psnr_db": float(baseline_psnr[local_index]),
                        "direct_psnr_db": float(direct_psnr[local_index]),
                        "iterative_psnr_db": float(denoised_psnr[local_index]),
                        "psnr_gain_db": float(
                            denoised_psnr[local_index] - baseline_psnr[local_index]
                        ),
                        "channel_ber": raw,
                        "direct_denoised_ber": direct_value,
                        "iterative_denoised_ber": final,
                        "direct_ber_correction_rate": (
                            (raw - direct_value) / raw if raw > 0 else 0.0
                        ),
                        "iterative_ber_correction_rate": (
                            (raw - final) / raw if raw > 0 else 0.0
                        ),
                    }
                    per_image_rows.append(row)
                    preview_rows.append(
                        {
                            "snr_db": snr,
                            "channel_ber": raw,
                            "denoised_ber": direct_value,
                            "ber_correction_rate": row["direct_ber_correction_rate"],
                            "baseline_psnr": row["baseline_psnr_db"],
                            "denoised_psnr": row["direct_psnr_db"],
                        }
                    )
                    global_index = processed + local_index
                    if global_index < args.save_images:
                        sample_dir = (
                            args.output_dir
                            / "images"
                            / f"eval_snr_{safe_snr_name(snr)}"
                        )
                        stem = Path(filename).stem
                        save_tensor_image(images[local_index], sample_dir / f"{stem}_gt.png")
                        save_tensor_image(
                            baseline_images[local_index],
                            sample_dir / f"{stem}_raw_qam.png",
                        )
                        save_tensor_image(
                            direct_images[local_index],
                            sample_dir / f"{stem}_bfm_direct.png",
                        )
                        save_tensor_image(
                            denoised_images[local_index],
                            sample_dir / f"{stem}_bfm_iterative.png",
                        )
                if batch_index == 0:
                    save_bfm_preview(
                        args.output_dir
                        / "previews"
                        / f"eval_snr_{safe_snr_name(snr)}.png",
                        images[: args.preview_images],
                        baseline_images[: args.preview_images],
                        direct_images[: args.preview_images],
                        preview_rows[: args.preview_images],
                    )
            processed += images.shape[0]
            print(f"evaluated {processed}/100 images", flush=True)

    summary_rows = []
    for snr_value in args.snrs:
        snr = float(snr_value)
        values = accumulators[snr]
        raw = values["channel_errors"] / values["bit_count"]
        direct_value = values["direct_errors"] / values["bit_count"]
        final = values["denoised_errors"] / values["bit_count"]
        theory = float(general_m_qam_ber(torch.tensor([snr]), args.qam_order)[0])
        row = {
            "eval_ebn0_db": snr,
            "path_start": sampler.path_step_from_snr(snr),
            "theoretical_channel_ber": theory,
            "measured_channel_ber": raw,
            "direct_denoised_ber": direct_value,
            "iterative_denoised_ber": final,
            "direct_ber_correction_rate": (
                (raw - direct_value) / raw if raw > 0 else 0.0
            ),
            "iterative_ber_correction_rate": (
                (raw - final) / raw if raw > 0 else 0.0
            ),
            "clean_psnr_mean_db": float(np.mean(values["clean_psnr"])),
            "raw_qam_psnr_mean_db": float(np.mean(values["baseline_psnr"])),
            "raw_qam_psnr_std_db": float(np.std(values["baseline_psnr"])),
            "bfm_direct_psnr_mean_db": float(np.mean(values["direct_psnr"])),
            "bfm_direct_psnr_std_db": float(np.std(values["direct_psnr"])),
            "bfm_iterative_psnr_mean_db": float(np.mean(values["denoised_psnr"])),
            "bfm_iterative_psnr_std_db": float(np.std(values["denoised_psnr"])),
            "direct_psnr_gain_db": float(
                np.mean(values["direct_psnr"])
                - np.mean(values["baseline_psnr"])
            ),
            "iterative_psnr_gain_db": float(
                np.mean(values["denoised_psnr"])
                - np.mean(values["baseline_psnr"])
            ),
            "noise_sigma_per_real_dim": ebn0_db_to_noise_sigma(
                snr, args.qam_order
            ),
            "num_images": processed,
            "bits_evaluated": values["bit_count"],
        }
        summary_rows.append(row)
        print(
            f"Eb/N0={snr:g} dB | BER raw/direct/iter={raw:.6f}/"
            f"{direct_value:.6f}/{final:.6f} | PSNR raw/direct/iter="
            f"{row['raw_qam_psnr_mean_db']:.3f}/"
            f"{row['bfm_direct_psnr_mean_db']:.3f}/"
            f"{row['bfm_iterative_psnr_mean_db']:.3f} dB",
            flush=True,
        )

    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_payload.get("step"),
        "source_initialization": checkpoint_payload.get("source_initialization"),
        "test_manifest": str(args.manifest.resolve()),
        "num_test_images": processed,
        "qam_order": args.qam_order,
        "axis": "Eb/N0 (dB)",
        "snrs": args.snrs,
        "paired_channel_seed_with_jscc": True,
        "direct_ber_definition": (
            "Hamming(clean frozen-BAE bits, thresholded one-pass X0 logits)"
        ),
        "iterative_ber_definition": (
            "Hamming(clean frozen-BAE bits, expectation-consistent sampler output)"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
