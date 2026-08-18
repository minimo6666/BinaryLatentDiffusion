#!/usr/bin/env python3
"""Regression checks for the canonical Gray-QAM Eb/N0 channel."""

import math

import torch

from utils.m_qam_awgn_util import (
    ebn0_db_to_esn0_db,
    ebn0_db_to_noise_variance,
    make_code_noise_single,
)
from utils.qam_utils import Eb_No_dB_to_sigma_in_M_QAM, general_m_qam_ber


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert abs(ebn0_db_to_esn0_db(0.0, 16) - 10.0 * math.log10(4)) < 1e-12
    assert abs(ebn0_db_to_noise_variance(3.0, 16) - 1 / (8 * 10**0.3)) < 1e-12

    sigma_unit = Eb_No_dB_to_sigma_in_M_QAM(3.0, 16, normalize=True)
    assert abs(float(sigma_unit.square()) - ebn0_db_to_noise_variance(3.0, 16)) < 1e-7

    n_bits = 2_000_000
    for snr in (-10.0, -5.0, 0.0, 5.0, 10.0):
        generator = torch.Generator(device=device).manual_seed(81018 + int(snr * 10))
        clean = torch.randint(0, 2, (n_bits,), device=device, dtype=torch.float32)
        received = make_code_noise_single(
            clean, snr, qam_order=16, device=device, generator=generator
        )
        measured = float((received != clean).float().mean())
        theory = float(general_m_qam_ber(torch.tensor([snr]), 16)[0])
        standard_error = math.sqrt(theory * (1.0 - theory) / n_bits)
        assert abs(measured - theory) <= 5.0 * standard_error + 2.0 / n_bits, (
            snr,
            theory,
            measured,
        )
        print(f"Eb/N0={snr:5.1f} dB theory={theory:.8f} measured={measured:.8f}")


if __name__ == "__main__":
    main()
