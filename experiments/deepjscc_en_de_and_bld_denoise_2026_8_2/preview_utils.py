"""Fixed-image visual previews for BLD checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision

from common import decode_sequence, encode_clean_sequence
from utils.m_qam_awgn_util import make_code_noise_single


def _snr_tag(snr: float) -> str:
    value = f"{snr:g}".replace("-", "minus").replace(".", "p")
    return f"{value}_db"


def _psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate).square().flatten(1).mean(1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).mean().item())


def _timestep_for_snr(sampler, snr: float) -> tuple[int, float]:
    schedule = sampler.Eb_N0_dB_values.detach()
    clamped = min(max(float(snr), float(schedule.min())), float(schedule.max()))
    timestep = int(torch.abs(schedule - clamped).argmin().item())
    return timestep, float(schedule[timestep].item())


@torch.no_grad()
def save_step_preview(
    sampler,
    autoencoder,
    hparams,
    images: torch.Tensor,
    preview_root: Path,
    step: int,
    snrs: list[float],
    seed: int = 20260802,
) -> Path:
    """Save fixed GT/baseline/BLD images and three-row comparison grids."""
    preview_root = Path(preview_root)
    step_dir = preview_root / f"step_{int(step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    device = next(sampler.parameters()).device
    images = images.to(device, non_blocking=True)
    x0 = encode_clean_sequence(autoencoder, images)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    metrics = []
    sampler.eval()
    try:
        for snr_index, snr in enumerate(snrs):
            # Identical channel and reverse-process randomness for every checkpoint.
            current_seed = int(seed + 1000 * snr_index)
            torch.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
            noisy = make_code_noise_single(
                x0,
                eb_n0_db=float(snr),
                qam_order=int(hparams.qam_order),
                device=device,
            )
            baseline = decode_sequence(autoencoder, noisy, hparams).clamp(0.0, 1.0)
            timestep, schedule_snr = _timestep_for_snr(sampler, snr)
            denoised_code = sampler.sample(noisy, timestep)
            denoised = decode_sequence(
                autoencoder, denoised_code, hparams
            ).clamp(0.0, 1.0)

            snr_dir = step_dir / _snr_tag(float(snr))
            gt_dir = snr_dir / "gt"
            baseline_dir = snr_dir / "jscc25"
            denoised_dir = snr_dir / "jscc25_bld"
            for directory in (gt_dir, baseline_dir, denoised_dir):
                directory.mkdir(parents=True, exist_ok=True)

            for index in range(images.shape[0]):
                filename = f"{index:04d}.png"
                torchvision.utils.save_image(images[index], gt_dir / filename)
                torchvision.utils.save_image(baseline[index], baseline_dir / filename)
                torchvision.utils.save_image(denoised[index], denoised_dir / filename)

            # Row 1: GT; row 2: frozen JSCC-25; row 3: JSCC-25 + BLD.
            comparison = torch.cat((images, baseline, denoised), dim=0)
            torchvision.utils.save_image(
                comparison,
                snr_dir / "comparison_grid.png",
                nrow=images.shape[0],
                padding=4,
                pad_value=1.0,
            )
            metrics.append(
                {
                    "step": int(step),
                    "test_ebn0_db": float(snr),
                    "bld_schedule_ebn0_db": schedule_snr,
                    "jscc25_psnr": _psnr(images, baseline),
                    "jscc25_bld_psnr": _psnr(images, denoised),
                }
            )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_states)

    with (step_dir / "preview_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (preview_root / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "comparison_grid.png rows:\n"
            "  row 1 = original GT\n"
            "  row 2 = frozen JSCC-25 (same noisy latent)\n"
            "  row 3 = frozen JSCC-25 + BLD\n"
        )
    return step_dir
