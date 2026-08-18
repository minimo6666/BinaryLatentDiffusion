#!/usr/bin/env python3
"""Shared, auditable components for the fair C64 JSCC/BFM comparison."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hparams.defaults.binarygan_default import HparamsBinaryAE
from models.binaryae import BinaryAutoEncoder
from utils.m_qam_awgn_util import make_code_noise_single

DEFAULT_FFHQ_ROOT = Path("/mnt/data/0/mohao/data/ffhq/ffhq256_val_train")
DEFAULT_LIST_ROOT = PROJECT_ROOT / "data" / "dataset" / "ffhq"
DEFAULT_BASE_CHECKPOINT = Path(
    "/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64/"
    "saved_models/binaryae_ema_8100000.th"
)
CHANNEL_NAME = "Gray 16-QAM, unit Es, complex AWGN, hard decision, Eb/N0"
LATENT_SHAPE = (64, 16, 16)
BITS_PER_IMAGE = math.prod(LATENT_SHAPE)


class FFHQListDataset(Dataset):
    """Load exactly the filenames declared by the project's FFHQ split lists."""

    def __init__(self, image_root: Path, list_path: Path, names=None):
        self.image_root = Path(image_root)
        if names is None:
            names = [
                line.strip()
                for line in Path(list_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.names = list(names)
        self.paths = [self.image_root / name for name in self.names]
        if not self.paths:
            raise RuntimeError(f"Empty FFHQ split: {list_path}")
        missing = [str(path) for path in self.paths[:16] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing FFHQ files: {missing[:3]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (256, 256):
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.uint8).copy()
        return {"image": array, "name": self.names[index]}


def split_names(split: str, list_root: Path = DEFAULT_LIST_ROOT):
    filename = "ffhqtrain.txt" if split == "train" else "ffhqvalidation.txt"
    return [
        line.strip()
        for line in (Path(list_root) / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_disjoint_splits(list_root: Path = DEFAULT_LIST_ROOT):
    train = set(split_names("train", list_root))
    validation = set(split_names("validation", list_root))
    overlap = train.intersection(validation)
    if overlap:
        raise RuntimeError(f"FFHQ train/test leakage: {len(overlap)} overlapping files")
    return len(train), len(validation)


def make_dataset(
    split: str,
    ffhq_root: Path = DEFAULT_FFHQ_ROOT,
    list_root: Path = DEFAULT_LIST_ROOT,
    names=None,
):
    split_dir = "train" if split == "train" else "valid"
    return FFHQListDataset(
        Path(ffhq_root) / split_dir,
        Path(list_root) / (
            "ffhqtrain.txt" if split == "train" else "ffhqvalidation.txt"
        ),
        names=names,
    )


def make_test_manifest(path: Path, num_images: int = 100, seed: int = 20260818):
    names = split_names("validation")
    rng = random.Random(seed)
    selected = rng.sample(names, num_images)
    payload = {
        "split": "ffhqvalidation",
        "seed": seed,
        "num_images": num_images,
        "filenames": selected,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_test_manifest(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") != "ffhqvalidation":
        raise ValueError("Test manifest must use ffhqvalidation")
    return payload


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_hparams() -> SimpleNamespace:
    h = HparamsBinaryAE("ffhq")
    h.codebook_size = LATENT_SHAPE[0]
    h.latent_shape = [1, LATENT_SHAPE[1], LATENT_SHAPE[2]]
    h.deterministic = True
    h.norm_first = True
    h.use_tanh = False
    return h


def _extract_autoencoder_state(state):
    if "model" in state:
        state = state["model"]
    selected = {}
    for key, value in state.items():
        stripped = key[3:] if key.startswith("ae.") else key
        if stripped.startswith(("encoder.", "quantize.", "generator.")):
            selected[stripped] = value
    return selected


def load_autoencoder(checkpoint: Path, device: torch.device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = BinaryAutoEncoder(build_hparams())
    raw = torch.load(checkpoint, map_location="cpu")
    state = _extract_autoencoder_state(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("generator.logsigma")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    return model


def image_batch(batch, device):
    images = batch["image"].to(device, non_blocking=True).float() / 255.0
    return images.permute(0, 3, 1, 2).contiguous()


def encode_probabilities_and_bits(model, images):
    hidden = model.encoder(images)
    probabilities = model.quantize.proj(hidden)
    bits = (probabilities > 0.5).to(probabilities.dtype)
    if tuple(bits.shape[1:]) != LATENT_SHAPE:
        raise RuntimeError(f"Expected latent {LATENT_SHAPE}, got {tuple(bits.shape[1:])}")
    return probabilities, bits


def decode_bits(model, bits, straight_through_source=None):
    if straight_through_source is not None:
        bits = bits.detach() + straight_through_source - straight_through_source.detach()
    quant = torch.einsum("b n h w, n d -> b d h w", bits, model.quantize.embed.weight)
    return model.generator(quant)


def transmit_bits(bits, ebn0_db, qam_order=16, generator=None):
    return make_code_noise_single(
        bits,
        eb_n0_db=ebn0_db,
        qam_order=qam_order,
        device=bits.device,
        generator=generator,
    )


def jscc_forward(model, images, ebn0_db, qam_order=16, generator=None):
    probabilities, clean_bits = encode_probabilities_and_bits(model, images)
    received_bits = transmit_bits(
        clean_bits.detach(), ebn0_db, qam_order=qam_order, generator=generator
    )
    output = decode_bits(model, received_bits, straight_through_source=probabilities)
    return output, clean_bits.detach(), received_bits.detach()


@torch.no_grad()
def clean_reconstruction(model, images):
    _, bits = encode_probabilities_and_bits(model, images)
    return decode_bits(model, bits), bits


@torch.no_grad()
def cycle_bits(model, images):
    _, bits = encode_probabilities_and_bits(model, images)
    return bits


def per_image_psnr(reference, estimate):
    mse = (reference.float() - estimate.float()).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def bit_error_rate(reference, estimate):
    return (reference != estimate).float().flatten(1).mean(1)


def safe_snr_name(value):
    value = float(value)
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def save_checkpoint(path, model, optimizer, step, train_snr, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "train_ebn0_db": float(train_snr),
        "qam_order": int(args.qam_order),
        "channel": CHANNEL_NAME,
        "latent_shape": list(LATENT_SHAPE),
        "bits_per_image": BITS_PER_IMAGE,
        "args": vars(args),
    }
    torch.save(payload, path)


def save_tensor_image(tensor, path):
    array = (
        tensor.detach().float().clamp(0, 1).mul(255).round()
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_annotated_preview(path, images, clean, noisy, rows):
    """Save columns input / clean AE / channel output with per-image metrics."""
    tensors = [images, clean, noisy]
    batch = min(images.shape[0], len(rows))
    width, height = 3 * 256, batch * 286
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headings = ("ground truth", "clean reconstruction", "after QAM + JSCC")
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
            f"cycle {row['cycle_ber']:.4f} | PSNR clean {row['clean_psnr']:.2f} -> "
            f"noisy {row['noisy_psnr']:.2f} dB"
        )
        draw.text((6, top + 259), label, fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
