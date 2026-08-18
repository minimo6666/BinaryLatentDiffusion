#!/usr/bin/env python3
"""Fine-tune the LSUN expectation-consistent BFM on FFHQ physical QAM pairs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_SOURCE_CHECKPOINT,
    EXPERIMENT_DIR,
    bit_error_rate,
    build_hparams,
    code_to_sequence,
    create_sampler,
    decode_sequence,
    image_batch,
    load_autoencoder,
    load_bfm_checkpoint,
    load_source_weights,
    load_test_manifest,
    make_dataset,
    per_image_psnr,
    save_bfm_checkpoint,
    save_bfm_preview,
    seed_everything,
    transmit_bits,
)
from experiments.different_snr_jscc.common import assert_disjoint_splits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qam-order", type=int, default=16)
    parser.add_argument("--snr-min", type=float, default=-15)
    parser.add_argument("--snr-max", type=float, default=20)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--preview-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--preview-snr", type=float, default=0)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--ema-beta", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model, model, beta):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - beta)


@torch.no_grad()
def validation_preview(sampler, autoencoder, loader, args, step, device):
    sampler.eval()
    batch = next(iter(loader))
    images = image_batch(batch, device)[: args.preview_images]
    clean_code = autoencoder(images, code_only=True).detach()
    clean_bits = code_to_sequence(clean_code)
    generator = torch.Generator(device=device).manual_seed(args.seed + step)
    received = transmit_bits(
        clean_bits, args.preview_snr, args.qam_order, generator=generator
    )
    path_start = sampler.path_step_from_snr(args.preview_snr)
    denoised = sampler.sample_from_channel(
        received,
        path_start,
        temp=args.temperature,
        sample_steps=max(1, path_start),
    )
    baseline_images = decode_sequence(autoencoder, received).clamp(0, 1)
    denoised_images = decode_sequence(autoencoder, denoised).clamp(0, 1)
    raw_ber = bit_error_rate(clean_bits, received)
    final_ber = bit_error_rate(clean_bits, denoised)
    baseline_psnr = per_image_psnr(images, baseline_images)
    denoised_psnr = per_image_psnr(images, denoised_images)
    rows = []
    for index in range(images.shape[0]):
        raw = float(raw_ber[index])
        final = float(final_ber[index])
        rows.append(
            {
                "snr_db": args.preview_snr,
                "channel_ber": raw,
                "denoised_ber": final,
                "ber_correction_rate": (raw - final) / raw if raw > 0 else 0.0,
                "baseline_psnr": float(baseline_psnr[index]),
                "denoised_psnr": float(denoised_psnr[index]),
            }
        )
    save_bfm_preview(
        args.output_dir / "previews" / f"step_{step:06d}.png",
        images,
        baseline_images,
        denoised_images,
        rows,
    )
    summary = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in (
            "channel_ber",
            "denoised_ber",
            "ber_correction_rate",
            "baseline_psnr",
            "denoised_psnr",
        )
    }
    summary["step"] = step
    summary["eval_ebn0_db"] = args.preview_snr
    with (args.output_dir / "validation_history.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(summary) + "\n")
    sampler.train()
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_count, test_count = assert_disjoint_splits()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    hparams = build_hparams(
        qam_order=args.qam_order,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
    )
    sampler = create_sampler(hparams, device)
    source_metadata = load_source_weights(sampler, args.source_checkpoint)
    autoencoder = load_autoencoder(args.ae_checkpoint, device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        sampler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    start_step = 0
    latest = args.output_dir / "checkpoints" / "latest.pt"
    resume = args.resume if args.resume is not None else (latest if latest.is_file() else None)
    if resume is not None:
        payload = load_bfm_checkpoint(sampler, resume)
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        source_metadata = payload.get("source_initialization", source_metadata)
        print(f"Resumed {resume} at step {start_step}", flush=True)

    ema_sampler = copy.deepcopy(sampler).eval()
    for parameter in ema_sampler.parameters():
        parameter.requires_grad_(False)

    train_dataset = make_dataset("train")
    manifest = load_test_manifest(args.manifest)
    validation_dataset = make_dataset(
        "validation", names=manifest["filenames"][: args.preview_images]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.preview_images,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
    )
    train_iterator = iter(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    sampler.train()

    configuration = {
        **vars(args),
        "source_checkpoint": str(args.source_checkpoint),
        "ae_checkpoint": str(args.ae_checkpoint),
        "manifest": str(args.manifest),
        "source_migration": source_metadata,
        "train_split_size": train_count,
        "test_split_size": test_count,
        "training_observation": "physical Gray-QAM/AWGN/hard demapping",
        "reverse_bridge": "expectation-consistent BSC marginal approximation",
        "hparams": vars(hparams),
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"Physical-QAM BFM | FFHQ train/test={train_count}/{test_count} | "
        f"Eb/N0 path={args.snr_max:g}->{args.snr_min:g} dB | source={source_metadata}",
        flush=True,
    )

    metrics_path = args.output_dir / "train_metrics.csv"
    fields = [
        "step",
        "loss",
        "bce_loss",
        "semantic_loss",
        "physical_channel_ber",
        "theoretical_ber",
        "mean_ebn0_db",
        "seconds",
    ]
    handle = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        writer.writeheader()

    window = {key: 0.0 for key in fields[1:-1]}
    window_count = 0
    started = time.time()
    try:
        for step in range(start_step + 1, args.train_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            images = image_batch(batch, device)
            with torch.no_grad():
                clean_bits = code_to_sequence(
                    autoencoder(images, code_only=True).detach()
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                stats = sampler(clean_bits, gt_img=images, ae=autoencoder)
                loss = stats["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_sampler, sampler, args.ema_beta)

            values = {
                "loss": float(stats["loss"].detach()),
                "bce_loss": float(stats["bce_loss"].detach()),
                "semantic_loss": float(stats.get("semantic_loss", 0.0)),
                "physical_channel_ber": float(stats["physical_channel_ber"]),
                "theoretical_ber": float(stats["theoretical_ber"]),
                "mean_ebn0_db": float(stats["physical_ebn0_db"]),
            }
            for key, value in values.items():
                window[key] += value
            window_count += 1

            if step % args.log_every == 0 or step == 1:
                row = {"step": step}
                row.update({key: value / window_count for key, value in window.items()})
                row["seconds"] = time.time() - started
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:06d} loss={row['loss']:.5f} BCE={row['bce_loss']:.5f} "
                    f"semantic={row['semantic_loss']:.5f} Eb/N0={row['mean_ebn0_db']:.2f} dB "
                    f"BER physical/theory={row['physical_channel_ber']:.5f}/"
                    f"{row['theoretical_ber']:.5f} time={row['seconds']:.1f}s",
                    flush=True,
                )
                window = {key: 0.0 for key in fields[1:-1]}
                window_count = 0
                started = time.time()

            if step % args.preview_every == 0 or step == 1:
                preview = validation_preview(
                    ema_sampler, autoencoder, validation_loader, args, step, device
                )
                print(
                    f"test-preview Eb/N0={args.preview_snr:g} dB | "
                    f"BER={preview['channel_ber']:.5f}->{preview['denoised_ber']:.5f} | "
                    f"PSNR={preview['baseline_psnr']:.2f}->{preview['denoised_psnr']:.2f} dB",
                    flush=True,
                )

            if step % args.save_every == 0:
                save_bfm_checkpoint(
                    latest, sampler, optimizer, step, args, source_metadata
                )
                save_bfm_checkpoint(
                    args.output_dir / "checkpoints" / f"bfm_qam_ema_{step:06d}.pt",
                    ema_sampler,
                    None,
                    step,
                    args,
                    source_metadata,
                )

        final = args.output_dir / "checkpoints" / "bfm_qam_ema_final.pt"
        save_bfm_checkpoint(
            final, ema_sampler, None, args.train_steps, args, source_metadata
        )
        save_bfm_checkpoint(
            latest, sampler, optimizer, args.train_steps, args, source_metadata
        )
        print(f"Training complete: {final}", flush=True)
    finally:
        handle.close()


if __name__ == "__main__":
    main()
