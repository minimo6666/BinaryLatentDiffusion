#!/usr/bin/env python3
"""Fail-fast audit proving JSCC and BFM use one identical physical channel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from common import (
    BITS_PER_IMAGE,
    CHANNEL_NAME,
    EXPERIMENT_DIR,
    LATENT_SHAPE,
    assert_disjoint_splits,
    load_test_manifest,
    transmit_bits as jscc_transmit_bits,
)
from experiments.bfm_expetation_consistent_2026_8_18.common import (
    DEFAULT_MANIFEST,
    transmit_bits as bfm_transmit_bits,
)
from utils.m_qam_awgn_util import (
    ebn0_db_to_esn0_db,
    ebn0_db_to_noise_sigma,
)
from utils.qam_utils import general_m_qam_ber

SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_count, test_count = assert_disjoint_splits()
    manifest = load_test_manifest(DEFAULT_MANIFEST)
    if len(manifest["filenames"]) != 100:
        raise RuntimeError("The shared test manifest must contain exactly 100 images")
    if BITS_PER_IMAGE != 64 * 16 * 16:
        raise RuntimeError("Unexpected C64 latent bitrate")

    n_bits = 4_000_000
    clean = torch.randint(
        0, 2, (n_bits,), generator=torch.Generator().manual_seed(81018)
    ).float().to(device)
    rows = []
    for index, snr in enumerate(SNRS):
        seed = 20260818 + index * 100_003
        generator_a = torch.Generator(device=device).manual_seed(seed)
        generator_b = torch.Generator(device=device).manual_seed(seed)
        jscc_received = jscc_transmit_bits(
            clean, snr, qam_order=16, generator=generator_a
        )
        bfm_received = bfm_transmit_bits(
            clean, snr, qam_order=16, generator=generator_b
        )
        if not torch.equal(jscc_received, bfm_received):
            raise RuntimeError(f"JSCC/BFM channel mismatch at Eb/N0={snr}")
        measured = float((clean != jscc_received).float().mean())
        theory = float(general_m_qam_ber(torch.tensor([snr]), 16)[0])
        standard_error = math.sqrt(theory * (1.0 - theory) / n_bits)
        if abs(measured - theory) > 5 * standard_error + 2 / n_bits:
            raise RuntimeError(
                f"BER theory mismatch at {snr}: measured={measured}, theory={theory}"
            )
        rows.append(
            {
                "ebn0_db": snr,
                "esn0_db": ebn0_db_to_esn0_db(snr, 16),
                "sigma_per_real_dimension": ebn0_db_to_noise_sigma(snr, 16),
                "theoretical_ber_uniform_bits": theory,
                "measured_ber_uniform_bits": measured,
                "jscc_bfm_received_bits_exactly_equal": True,
            }
        )
        print(
            f"Eb/N0={snr:>3} dB sigma={rows[-1]['sigma_per_real_dimension']:.8f} "
            f"BER theory/measured={theory:.8f}/{measured:.8f} exact_match=yes"
        )

    report = {
        "status": "PASS",
        "channel": CHANNEL_NAME,
        "qam_order": 16,
        "latent_shape": list(LATENT_SHAPE),
        "bits_per_image": BITS_PER_IMAGE,
        "ffhq_train_images": train_count,
        "ffhq_test_images": test_count,
        "split_overlap": 0,
        "shared_test_manifest": str(Path(DEFAULT_MANIFEST).resolve()),
        "shared_test_images": len(manifest["filenames"]),
        "channel_implementation": (
            "utils.m_qam_awgn_util.make_code_noise_single"
        ),
        "rows": rows,
    }
    output = EXPERIMENT_DIR / "fairness_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bfm_output = (
        EXPERIMENT_DIR.parent
        / "bfm_expetation_consistent_2026_8_18"
        / "fairness_audit.json"
    )
    bfm_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
