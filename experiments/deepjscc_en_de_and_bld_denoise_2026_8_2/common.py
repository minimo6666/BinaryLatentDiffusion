"""Shared utilities for the frozen DeepJSCC-25 + BLD experiment."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BLD_ROOT = EXPERIMENT_ROOT.parents[1]
WORKSPACE_ROOT = BLD_ROOT.parent
if str(BLD_ROOT) not in sys.path:
    sys.path.insert(0, str(BLD_ROOT))

DEFAULT_JSCC_DIR = Path(
    "/mnt/data/0/mohao/Project/BLD_DSC/logs/"
    "different_snrs_jscc_en_de_ffhq/snr_25"
)
DEFAULT_JSCC_CHECKPOINT = (
    DEFAULT_JSCC_DIR / "saved_models" / "binaryae_ema_100000.th"
)
DEFAULT_REFERENCE_PSNR = Path(
    "/mnt/data/0/mohao/Project/BLD_DSC/logs/"
    "different_snrs_jscc_en_de_ffhq/"
    "en_de_train_under_snr_25/psnr_summary.txt"
)
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_hparams(config: Dict[str, Any]) -> SimpleNamespace:
    """Build the exact AE/BLD construction parameters used by this experiment."""
    defaults: Dict[str, Any] = {
        # Frozen JSCC-25 binary autoencoder architecture.
        "n_channels": 3,
        "nf": 128,
        "res_blocks": 2,
        "codebook_size": 96,
        "emb_dim": 256,
        "ch_mult": [1, 1, 2, 4],
        "img_size": 256,
        "attn_resolutions": [16],
        "quantizer": "binary",
        "beta": 0.25,
        "deterministic": True,
        "use_tanh": False,
        "gen_mul": 1.0,
        "norm_first": True,
        # BLD schedule and loss.
        "sampler": "bld_dsc",
        "dataset": "ffhq",
        "batch_size": 2,
        "latent_shape": [1, 32, 32],
        "block_size": 1024,
        "total_steps": 64,
        "sample_steps": 64,
        "snr_range": [0.0, 15.0],
        "qam_order": 16,
        "loss_type": None,
        "mask_schedule": None,
        "loss_final": "mean",
        "p_flip": True,
        "focal": 0.0,
        "aux": 0.0,
        "use_softmax": False,
        "guidance": False,
        # Memory-conscious trial Transformer. These are saved in every checkpoint.
        "bert_n_emb": 384,
        "bert_n_head": 6,
        "bert_n_layers": 8,
        "attn_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "drop_path": 0.0,
        "cross": False,
        "time_cond_mode": "adaln",
    }
    defaults.update(config)
    if defaults["block_size"] != defaults["latent_shape"][1] * defaults["latent_shape"][2]:
        raise ValueError("block_size must equal latent height * latent width")
    if defaults["bert_n_emb"] % defaults["bert_n_head"] != 0:
        raise ValueError("bert_n_emb must be divisible by bert_n_head")
    return SimpleNamespace(**defaults)


def model_config_from_args(args: Any) -> Dict[str, Any]:
    return {
        "qam_order": int(args.qam_order),
        "snr_range": [float(args.snr_min), float(args.snr_max)],
        "bert_n_emb": int(args.hidden_size),
        "bert_n_head": int(args.num_heads),
        "bert_n_layers": int(args.num_layers),
        "time_cond_mode": "adaln",
    }


def load_frozen_jscc(
    checkpoint_path: Path | str,
    hparams: SimpleNamespace,
    device: torch.device,
):
    """Load JSCC weights into the channel-free AE class to expose clean X0."""
    from models.binaryae import BinaryAutoEncoder

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JSCC checkpoint not found: {checkpoint_path}")

    model = BinaryAutoEncoder(hparams)
    full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    components = ("encoder", "quantize", "generator")
    ae_state = {}
    for key, value in full_state.items():
        if not key.startswith("ae."):
            continue
        stripped = key[3:]
        if stripped.startswith(components):
            ae_state[stripped] = value
    missing, unexpected = model.load_state_dict(ae_state, strict=False)
    missing = [key for key in missing if not key.startswith("generator.logsigma")]
    if missing or unexpected:
        raise RuntimeError(
            f"JSCC checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    del full_state, ae_state
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_ffhq_dataset(split: str):
    from ffhq_dataset import FFHQListDataset

    split_root = Path("/mnt/data/0/mohao/data/ffhq/ffhq256_val_train")
    list_root = BLD_ROOT / "data" / "dataset" / "ffhq"
    if split == "train":
        return FFHQListDataset(split_root / "train", list_root / "ffhqtrain.txt")
    if split in {"validation", "val"}:
        return FFHQListDataset(
            split_root / "valid", list_root / "ffhqvalidation.txt"
        )
    raise ValueError(f"Unknown FFHQ split: {split}")


def image_batch_to_float(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    images = data["image"].float() / 255.0
    return images.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def encode_clean_sequence(autoencoder, images: torch.Tensor) -> torch.Tensor:
    """Return pre-channel clean binary X0 with shape [B, 1024, 96]."""
    code = autoencoder(images, code_only=True).detach()
    batch, channels, height, width = code.shape
    return code.view(batch, channels, height * width).permute(0, 2, 1).contiguous()


def decode_sequence(autoencoder, sequence: torch.Tensor, hparams: SimpleNamespace) -> torch.Tensor:
    batch = sequence.shape[0]
    code = sequence.permute(0, 2, 1).contiguous().view(
        batch,
        hparams.codebook_size,
        hparams.latent_shape[1],
        hparams.latent_shape[2],
    )
    images, _, _ = autoencoder(None, code=code.float())
    return images


def save_json(path: Path | str, value: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_snrs(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one evaluation SNR is required")
    return result


def cycle(loader: Iterable[Any]):
    while True:
        yield from loader
