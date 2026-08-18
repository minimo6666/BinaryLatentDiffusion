#!/usr/bin/env python3
"""Evaluate one fixed-SNR JSCC checkpoint on 100 held-out FFHQ images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    CHANNEL_NAME,
    EXPERIMENT_DIR,
    bit_error_rate,
    cycle_bits,
    decode_bits,
    encode_probabilities_and_bits,
    image_batch,
    load_autoencoder,
    load_test_manifest,
    make_dataset,
    per_image_psnr,
    safe_snr_name,
    save_annotated_preview,
    save_tensor_image,
    seed_everything,
    transmit_bits,
)
from utils.m_qam_awgn_util import ebn0_db_to_noise_sigma
from utils.qam_utils import general_m_qam_ber

DEFAULT_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-snr", type=float, required=True)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument(
        "--manifest", type=Path, default=EXPERIMENT_DIR / "test_manifest_100.json"
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--qam-order", type=int, default=16)
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
        raise RuntimeError(f"Expected 100 held-out images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_autoencoder(args.checkpoint, device).eval()
    train_name = safe_snr_name(args.train_snr)
    result_dir = args.output_root / f"train_snr_{train_name}"
    image_root = result_dir / "images"
    result_dir.mkdir(parents=True, exist_ok=True)

    accumulators = {
        float(snr): {
            "clean_psnr": [],
            "noisy_psnr": [],
            "channel_errors": 0,
            "cycle_errors": 0,
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
            probabilities, clean_bits = encode_probabilities_and_bits(model, images)
            clean_images = decode_bits(model, clean_bits).clamp(0, 1)
            clean_psnr = per_image_psnr(images, clean_images)

            for snr_index, snr in enumerate(args.snrs):
                snr = float(snr)
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_003 * snr_index + batch_index
                )
                received_bits = transmit_bits(
                    clean_bits, snr, args.qam_order, generator=generator
                )
                output = decode_bits(model, received_bits).clamp(0, 1)
                recovered_cycle_bits = cycle_bits(model, output)
                noisy_psnr = per_image_psnr(images, output)
                channel_ber = bit_error_rate(clean_bits, received_bits)
                cycle_ber = bit_error_rate(clean_bits, recovered_cycle_bits)

                metrics = accumulators[snr]
                metrics["clean_psnr"].extend(clean_psnr.cpu().tolist())
                metrics["noisy_psnr"].extend(noisy_psnr.cpu().tolist())
                metrics["channel_errors"] += int(
                    (clean_bits != received_bits).sum().item()
                )
                metrics["cycle_errors"] += int(
                    (clean_bits != recovered_cycle_bits).sum().item()
                )
                metrics["bit_count"] += clean_bits.numel()

                preview_rows = []
                for local_index, filename in enumerate(names):
                    raw = float(channel_ber[local_index])
                    post = float(cycle_ber[local_index])
                    row = {
                        "filename": filename,
                        "train_ebn0_db": args.train_snr,
                        "eval_ebn0_db": snr,
                        "clean_psnr_db": float(clean_psnr[local_index]),
                        "noisy_psnr_db": float(noisy_psnr[local_index]),
                        "psnr_drop_db": float(
                            noisy_psnr[local_index] - clean_psnr[local_index]
                        ),
                        "channel_ber": raw,
                        "cycle_ber": post,
                        "cycle_ber_correction_rate": (
                            (raw - post) / raw if raw > 0 else 0.0
                        ),
                    }
                    per_image_rows.append(row)
                    preview_rows.append(
                        {
                            "snr_db": snr,
                            "channel_ber": raw,
                            "cycle_ber": post,
                            "clean_psnr": row["clean_psnr_db"],
                            "noisy_psnr": row["noisy_psnr_db"],
                        }
                    )
                    global_index = processed + local_index
                    if global_index < args.save_images:
                        sample_dir = image_root / f"eval_snr_{safe_snr_name(snr)}"
                        stem = Path(filename).stem
                        save_tensor_image(images[local_index], sample_dir / f"{stem}_gt.png")
                        save_tensor_image(
                            clean_images[local_index], sample_dir / f"{stem}_clean.png"
                        )
                        save_tensor_image(
                            output[local_index], sample_dir / f"{stem}_jscc.png"
                        )

                if batch_index == 0:
                    save_annotated_preview(
                        result_dir
                        / "previews"
                        / f"eval_snr_{safe_snr_name(snr)}.png",
                        images[: args.preview_images],
                        clean_images[: args.preview_images],
                        output[: args.preview_images],
                        preview_rows[: args.preview_images],
                    )
            processed += images.shape[0]
            print(f"evaluated {processed}/100 images", flush=True)

    summary_rows = []
    for snr in map(float, args.snrs):
        values = accumulators[snr]
        raw_ber = values["channel_errors"] / values["bit_count"]
        cycle_ber = values["cycle_errors"] / values["bit_count"]
        theory = float(general_m_qam_ber(torch.tensor([snr]), args.qam_order)[0])
        row = {
            "train_ebn0_db": args.train_snr,
            "eval_ebn0_db": snr,
            "theoretical_channel_ber": theory,
            "measured_channel_ber": raw_ber,
            "cycle_ber": cycle_ber,
            "cycle_ber_correction_rate": (
                (raw_ber - cycle_ber) / raw_ber if raw_ber > 0 else 0.0
            ),
            "clean_psnr_mean_db": float(np.mean(values["clean_psnr"])),
            "clean_psnr_std_db": float(np.std(values["clean_psnr"])),
            "noisy_psnr_mean_db": float(np.mean(values["noisy_psnr"])),
            "noisy_psnr_std_db": float(np.std(values["noisy_psnr"])),
            "psnr_change_from_clean_db": float(
                np.mean(values["noisy_psnr"]) - np.mean(values["clean_psnr"])
            ),
            "noise_sigma_per_real_dim": ebn0_db_to_noise_sigma(snr, args.qam_order),
            "num_images": processed,
            "bits_evaluated": values["bit_count"],
        }
        summary_rows.append(row)
        print(
            f"train={args.train_snr:g} eval={snr:g} dB | "
            f"BER theory/measured/cycle={theory:.6f}/{raw_ber:.6f}/{cycle_ber:.6f} | "
            f"PSNR clean/noisy={row['clean_psnr_mean_db']:.3f}/"
            f"{row['noisy_psnr_mean_db']:.3f} dB",
            flush=True,
        )

    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (result_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_ebn0_db": args.train_snr,
        "eval_ebn0_db": args.snrs,
        "channel": CHANNEL_NAME,
        "qam_order": args.qam_order,
        "num_test_images": processed,
        "test_manifest": str(args.manifest.resolve()),
        "cycle_ber_definition": (
            "Hamming(clean encoder bits, encoder(JSCC reconstructed image) bits). "
            "JSCC has no explicit corrected-bit output; this metric is not the same "
            "as the BFM post-denoising BER."
        ),
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
