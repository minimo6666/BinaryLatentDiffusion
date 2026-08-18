#!/usr/bin/env python3
"""Fine-tune one C64 FFHQ encoder/decoder at one fixed physical Eb/N0."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    BITS_PER_IMAGE,
    CHANNEL_NAME,
    DEFAULT_BASE_CHECKPOINT,
    EXPERIMENT_DIR,
    assert_disjoint_splits,
    bit_error_rate,
    clean_reconstruction,
    cycle_bits,
    image_batch,
    jscc_forward,
    load_autoencoder,
    load_test_manifest,
    make_dataset,
    make_test_manifest,
    per_image_psnr,
    safe_snr_name,
    save_annotated_preview,
    save_checkpoint,
    seed_everything,
)
from utils.qam_utils import general_m_qam_ber


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-snr", type=float, required=True, help="Eb/N0 in dB")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--qam-order", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--preview-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def validation_preview(model, loader, args, step, run_dir, device):
    model.eval()
    batch = next(iter(loader))
    images = image_batch(batch, device)[: args.preview_images]
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 1_000_003 * int(round(args.train_snr + 100)) + step
    )
    with torch.cuda.amp.autocast(enabled=not args.no_amp):
        output, clean_bits, received_bits = jscc_forward(
            model, images, args.train_snr, args.qam_order, generator
        )
        clean_images, _ = clean_reconstruction(model, images)
        recovered_cycle_bits = cycle_bits(model, output.clamp(0, 1))
    clean_psnr = per_image_psnr(images, clean_images.clamp(0, 1))
    noisy_psnr = per_image_psnr(images, output.clamp(0, 1))
    channel_ber = bit_error_rate(clean_bits, received_bits)
    cycle_ber = bit_error_rate(clean_bits, recovered_cycle_bits)
    rows = []
    for index in range(images.shape[0]):
        raw = float(channel_ber[index])
        post = float(cycle_ber[index])
        rows.append(
            {
                "snr_db": args.train_snr,
                "channel_ber": raw,
                "cycle_ber": post,
                "ber_correction_rate": (raw - post) / raw if raw > 0 else 0.0,
                "clean_psnr": float(clean_psnr[index]),
                "noisy_psnr": float(noisy_psnr[index]),
            }
        )
    save_annotated_preview(
        run_dir / "previews" / f"step_{step:06d}.png",
        images,
        clean_images,
        output,
        rows,
    )
    summary = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in (
            "channel_ber",
            "cycle_ber",
            "ber_correction_rate",
            "clean_psnr",
            "noisy_psnr",
        )
    }
    summary["step"] = step
    summary["train_ebn0_db"] = args.train_snr
    with (run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    model.train()
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for JSCC fine-tuning")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed + int(round(10 * args.train_snr)))

    train_count, validation_count = assert_disjoint_splits()
    manifest_path = EXPERIMENT_DIR / "test_manifest_100.json"
    if not manifest_path.exists():
        make_test_manifest(manifest_path, num_images=100, seed=args.seed)
    manifest = load_test_manifest(manifest_path)

    run_dir = args.output_root / f"train_snr_{safe_snr_name(args.train_snr)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configuration.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output_root": str(args.output_root),
                "base_checkpoint": str(args.base_checkpoint),
                "resume": str(args.resume) if args.resume else None,
                "channel": CHANNEL_NAME,
                "axis": "Eb/N0 (dB)",
                "latent_bits_per_image": BITS_PER_IMAGE,
                "train_split_size": train_count,
                "test_split_size": validation_count,
                "test_manifest": str(manifest_path),
                "objective": "mean squared image error (direct PSNR objective)",
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    model = load_autoencoder(args.base_checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    resume = args.resume
    latest = run_dir / "checkpoints" / "latest.pt"
    if resume is None and latest.is_file():
        resume = latest
    if resume is not None:
        payload = torch.load(resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        print(f"Resumed {resume} at step {start_step}", flush=True)

    train_dataset = make_dataset("train")
    validation_dataset = make_dataset(
        "validation", names=manifest["filenames"][: args.preview_images]
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
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
    model.train()

    theory = float(
        general_m_qam_ber(torch.tensor([args.train_snr]), args.qam_order)[0]
    )
    print(
        f"train Eb/N0={args.train_snr:g} dB | theoretical BER={theory:.8f} | "
        f"channel={CHANNEL_NAME} | bits/image={BITS_PER_IMAGE} | "
        f"train/test={train_count}/{validation_count}",
        flush=True,
    )

    metrics_path = run_dir / "train_metrics.csv"
    metric_fields = ["step", "loss_mse", "psnr_db", "channel_ber", "seconds"]
    write_header = not metrics_path.exists() or start_step == 0
    metrics_handle = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    if write_header:
        writer.writeheader()

    window_loss = 0.0
    window_psnr = 0.0
    window_ber = 0.0
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
            if torch.rand((), device=device) < 0.5:
                images = torch.flip(images, dims=(-1,))

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                output, clean_bits, received_bits = jscc_forward(
                    model, images, args.train_snr, args.qam_order
                )
                loss = F.mse_loss(output, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                psnr = per_image_psnr(images, output.clamp(0, 1)).mean()
                ber = bit_error_rate(clean_bits, received_bits).mean()
            window_loss += float(loss)
            window_psnr += float(psnr)
            window_ber += float(ber)
            window_count += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - started
                row = {
                    "step": step,
                    "loss_mse": window_loss / window_count,
                    "psnr_db": window_psnr / window_count,
                    "channel_ber": window_ber / window_count,
                    "seconds": elapsed,
                }
                writer.writerow(row)
                metrics_handle.flush()
                print(
                    f"step={step:06d} mse={row['loss_mse']:.6f} "
                    f"PSNR={row['psnr_db']:.3f} dB channel_BER={row['channel_ber']:.6f} "
                    f"theory_BER={theory:.6f} time={elapsed:.1f}s",
                    flush=True,
                )
                window_loss = window_psnr = window_ber = 0.0
                window_count = 0
                started = time.time()

            if step % args.preview_every == 0 or step == 1:
                summary = validation_preview(
                    model, validation_loader, args, step, run_dir, device
                )
                print(
                    "test-preview "
                    f"Eb/N0={args.train_snr:g} dB BER={summary['channel_ber']:.5f} "
                    f"cycle_BER={summary['cycle_ber']:.5f} "
                    f"PSNR={summary['clean_psnr']:.2f}->{summary['noisy_psnr']:.2f} dB",
                    flush=True,
                )

            if step % args.save_every == 0:
                save_checkpoint(
                    latest, model, optimizer, step, args.train_snr, args
                )

        final_path = (
            run_dir
            / "checkpoints"
            / f"jscc_train_snr_{safe_snr_name(args.train_snr)}_final.pt"
        )
        save_checkpoint(final_path, model, None, args.train_steps, args.train_snr, args)
        save_checkpoint(latest, model, optimizer, args.train_steps, args.train_snr, args)
        print(f"Training complete: {final_path}", flush=True)
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
