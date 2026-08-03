#!/usr/bin/env python3
"""Train BLD on clean latent codes produced by the frozen JSCC-25 encoder."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from bld_model import ExactQAMAWGNBinaryDiffusionDSC
from common import (
    DEFAULT_JSCC_CHECKPOINT,
    DEFAULT_RESULTS_DIR,
    build_hparams,
    cycle,
    encode_clean_sequence,
    image_batch_to_float,
    load_ffhq_dataset,
    load_frozen_jscc,
    model_config_from_args,
    parse_snrs,
    save_json,
    seed_everything,
)
from models.transformer import TransformerBD
from preview_utils import save_step_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jscc-checkpoint", type=Path, default=DEFAULT_JSCC_CHECKPOINT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--qam-order", type=int, default=16, choices=[4, 16, 64, 256])
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument("--preview-snrs", default="0,4,8,12")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.999) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def save_checkpoint(path: Path, sampler, ema_sampler, optimizer, scaler, step, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "config": config,
        "sampler": sampler.state_dict(),
        "ema_sampler": ema_sampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    seed_everything(args.seed)
    device = torch.device("cuda")
    amp_enabled = not args.no_amp

    model_config = model_config_from_args(args)
    hparams = build_hparams(model_config)
    result_dir = args.result_dir.resolve()
    checkpoint_dir = result_dir / "checkpoints"
    log_dir = result_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **model_config,
        "jscc_checkpoint": str(args.jscc_checkpoint.resolve()),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "channel": "exact normalized M-QAM + AWGN + hard demapping",
        "x0": "pre-channel clean binary code from frozen JSCC-25 encoder",
        "trainable": "BLD only; JSCC-25 encoder and decoder frozen",
    }
    save_json(result_dir / "run_config.json", run_config)

    print(f"Loading frozen JSCC-25: {args.jscc_checkpoint}")
    autoencoder = load_frozen_jscc(args.jscc_checkpoint, hparams, device)

    denoiser = TransformerBD(hparams).to(device)
    sampler = ExactQAMAWGNBinaryDiffusionDSC(
        hparams, denoiser, hparams.codebook_size
    ).to(device)
    ema_sampler = copy.deepcopy(sampler).eval()
    for parameter in ema_sampler.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        sampler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        sampler.load_state_dict(state["sampler"], strict=True)
        ema_sampler.load_state_dict(state["ema_sampler"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step = int(state["step"])
        print(f"Resumed from step {start_step}: {args.resume}")

    dataset = load_ffhq_dataset("train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    batches = cycle(loader)
    preview_images = None
    if not args.no_preview and args.preview_images > 0:
        preview_dataset = load_ffhq_dataset("validation")
        preview_loader = DataLoader(
            preview_dataset,
            batch_size=args.preview_images,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        preview_images = image_batch_to_float(next(iter(preview_loader)))[
            : args.preview_images
        ]
    metrics_path = log_dir / "train_metrics.jsonl"
    sampler.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for step in range(start_step + 1, args.train_steps + 1):
        images = image_batch_to_float(next(batches)).to(device, non_blocking=True)
        with torch.no_grad():
            x0 = encode_clean_sequence(autoencoder, images)

        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(1, args.warmup_steps)
        else:
            learning_rate = args.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            stats = sampler(x0, gt_img=images, ae=autoencoder)
            loss = stats["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema_sampler, sampler)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "bce_loss": float(stats["bce_loss"]),
                "semantic_loss": float(stats.get("semantic_loss", 0.0)),
                "accuracy": float(stats["acc"]),
                "channel_ber": float(stats["channel_ber"]),
                "snr_mean_db": float(stats["snr_mean_db"]),
                "grad_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step={step:06d} loss={record['loss']:.5f} "
                f"bce={record['bce_loss']:.5f} sem={record['semantic_loss']:.5f} "
                f"acc={record['accuracy']:.4f} ber={record['channel_ber']:.4f} "
                f"snr={record['snr_mean_db']:.2f} lr={learning_rate:.2e}"
            )

        if step % args.save_every == 0 or step == args.train_steps:
            numbered = checkpoint_dir / f"bld_step_{step:06d}.pt"
            save_checkpoint(
                numbered, sampler, ema_sampler, optimizer, scaler, step, run_config
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                sampler,
                ema_sampler,
                optimizer,
                scaler,
                step,
                run_config,
            )
            print(f"Saved checkpoint: {numbered}")
            if preview_images is not None:
                preview_dir = save_step_preview(
                    ema_sampler,
                    autoencoder,
                    hparams,
                    preview_images,
                    result_dir / "previews",
                    step,
                    parse_snrs(args.preview_snrs),
                    seed=args.seed,
                )
                sampler.train()
                print(f"Saved visual preview: {preview_dir}")


if __name__ == "__main__":
    main()
