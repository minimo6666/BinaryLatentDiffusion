#!/usr/bin/env python3
"""Shared configuration and I/O for the physical-QAM BFM experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bfm_expetation_consistent_2026_8_18.bfm_model import PhysicalQAMExpectationConsistentBFM
from experiments.different_snr_jscc.common import (
    BITS_PER_IMAGE,
    CHANNEL_NAME,
    DEFAULT_BASE_CHECKPOINT,
    LATENT_SHAPE,
    bit_error_rate,
    image_batch,
    load_autoencoder,
    load_test_manifest,
    make_dataset,
    per_image_psnr,
    safe_snr_name,
    save_tensor_image,
    seed_everything,
    transmit_bits,
)
from hparams.defaults.binarygan_default import HparamsBinaryAE
from hparams.defaults.sampler_defaults import HparamsBianryLatent
from models.transformer import TransformerBD

DEFAULT_SOURCE_CHECKPOINT = Path(
    "/mnt/data/b/mohao/Projects/UnifyBinaryFlowDiffusion/BinaryLatentDiffusion/"
    "experiments/bernoulli_flow_lsun_churches_256_expectation_consistent_aux0p1/"
    "saved_models/flow_lsun_ema_50000.th"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "different_snr_jscc" / "test_manifest_100.json"


def build_hparams(
    qam_order=16,
    snr_min=-15.0,
    snr_max=20.0,
    total_steps=64,
    batch_size=8,
):
    h = HparamsBinaryAE("ffhq")
    h.vqgan_batch_size = h.batch_size
    h.update(HparamsBianryLatent("ffhq"))
    h.update(
        sampler="bfm_qam_ec",
        dataset="ffhq",
        codebook_size=LATENT_SHAPE[0],
        latent_shape=[1, LATENT_SHAPE[1], LATENT_SHAPE[2]],
        block_size=LATENT_SHAPE[1] * LATENT_SHAPE[2],
        img_size=256,
        qam_order=int(qam_order),
        snr_range=[float(snr_min), float(snr_max)],
        total_steps=int(total_steps),
        sample_steps=int(total_steps),
        temp=0.9,
        batch_size=int(batch_size),
        loss_final="mean",
        p_flip=True,
        norm_first=True,
        aux=0.1,
        focal=0,
        use_softmax=False,
        guidance=False,
        cross=False,
        bert_n_emb=768,
        bert_n_head=12,
        bert_n_layers=24,
        time_cond_mode="token",
        x0_posterior_mode="expectation_consistent",
        aux_interval_mode="mixed",
    )
    return h


def create_sampler(h, device):
    denoiser = TransformerBD(h)
    model = PhysicalQAMExpectationConsistentBFM(h, denoiser, h.codebook_size)
    return model.to(device)


def load_source_weights(model, checkpoint=DEFAULT_SOURCE_CHECKPOINT):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"snr_values", "qam_ber", "channel_tau"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"BFM source mismatch: missing={bad_missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(state),
        "allowed_new_buffers": sorted(allowed_missing.intersection(missing)),
    }


def load_bfm_checkpoint(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload


def save_bfm_checkpoint(path, model, optimizer, step, args, source_metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": int(step),
            "channel": CHANNEL_NAME,
            "training_observation": "physical QAM/AWGN/hard-demapped bits",
            "latent_shape": list(LATENT_SHAPE),
            "bits_per_image": BITS_PER_IMAGE,
            "source_initialization": source_metadata,
            "args": vars(args),
        },
        path,
    )


def code_to_sequence(code):
    batch, channels, _, _ = code.shape
    return code.view(batch, channels, -1).permute(0, 2, 1).contiguous()


def sequence_to_code(sequence):
    batch = sequence.shape[0]
    return sequence.permute(0, 2, 1).contiguous().view(
        batch, LATENT_SHAPE[0], LATENT_SHAPE[1], LATENT_SHAPE[2]
    )


def decode_sequence(autoencoder, sequence):
    output, _, _ = autoencoder(None, code=sequence_to_code(sequence).float())
    return output


def save_bfm_preview(path, images, baseline, denoised, rows):
    tensors = [images, baseline, denoised]
    batch = min(images.shape[0], len(rows))
    width, height = 3 * 256, batch * 286
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headings = ("ground truth", "raw QAM bits decoded", "BFM denoised bits decoded")
    for column, heading in enumerate(headings):
        draw.text((column * 256 + 6, 4), heading, fill="black")
    for index in range(batch):
        top = index * 286 + 24
        for column, tensor_batch in enumerate(tensors):
            array = (
                tensor_batch[index].detach().float().clamp(0, 1).mul(255).round()
                .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            )
            canvas.paste(Image.fromarray(array), (column * 256, top))
        row = rows[index]
        label = (
            f"Eb/N0={row['snr_db']:.1f} dB | BER {row['channel_ber']:.4f} -> "
            f"{row['denoised_ber']:.4f} ({100*row['ber_correction_rate']:.1f}% corrected) | "
            f"PSNR {row['baseline_psnr']:.2f} -> {row['denoised_psnr']:.2f} dB"
        )
        draw.text((6, top + 259), label, fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
