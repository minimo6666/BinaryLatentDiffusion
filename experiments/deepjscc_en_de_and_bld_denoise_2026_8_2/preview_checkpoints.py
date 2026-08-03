#!/usr/bin/env python3
"""Backfill fixed visual previews for every saved 5000-step checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from bld_model import ExactQAMAWGNBinaryDiffusionDSC
from common import (
    DEFAULT_JSCC_CHECKPOINT,
    DEFAULT_RESULTS_DIR,
    build_hparams,
    image_batch_to_float,
    load_ffhq_dataset,
    load_frozen_jscc,
    parse_snrs,
    seed_everything,
)
from models.transformer import TransformerBD
from preview_utils import save_step_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_RESULTS_DIR / "checkpoints")
    parser.add_argument("--jscc-checkpoint", type=Path, default=DEFAULT_JSCC_CHECKPOINT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--snrs", default="0,4,8,12")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint previews")
    seed_everything(args.seed)
    device = torch.device("cuda")
    checkpoints = sorted(args.checkpoint_dir.glob("bld_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No numbered checkpoints in {args.checkpoint_dir}")

    first_state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    hparams = build_hparams(dict(first_state["config"]))
    autoencoder = load_frozen_jscc(args.jscc_checkpoint, hparams, device)
    denoiser = TransformerBD(hparams).to(device)
    sampler = ExactQAMAWGNBinaryDiffusionDSC(
        hparams, denoiser, hparams.codebook_size
    ).to(device).eval()

    dataset = load_ffhq_dataset("validation")
    loader = DataLoader(
        dataset,
        batch_size=args.num_images,
        sampler=SequentialSampler(dataset),
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    fixed_images = image_batch_to_float(next(iter(loader)))[: args.num_images]
    snrs = parse_snrs(args.snrs)
    preview_root = args.result_dir.resolve() / "previews"

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        state = first_state if checkpoint_index == 0 else torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        sampler.load_state_dict(state["ema_sampler"], strict=True)
        step = int(state["step"])
        output = save_step_preview(
            sampler,
            autoencoder,
            hparams,
            fixed_images,
            preview_root,
            step,
            snrs,
            seed=args.seed,
        )
        print(f"Saved visual preview for step {step}: {output}", flush=True)
        if checkpoint_index == 0:
            del first_state
        else:
            del state
        torch.cuda.empty_cache()

    print(f"All checkpoint previews saved to: {preview_root}")


if __name__ == "__main__":
    main()
