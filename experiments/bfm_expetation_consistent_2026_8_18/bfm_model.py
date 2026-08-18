#!/usr/bin/env python3
"""Expectation-consistent BFM whose training observations are physical QAM bits."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

LEGACY_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "bfm_expetation_consistent_temp_0.9_compare_to_diff_jscc"
if str(LEGACY_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_EXPERIMENT_DIR))
from binarylatent_flow_expectation_consistent_retrain import BinaryDiffusionFlowDecouple
from utils.m_qam_awgn_util import make_code_noise_single


class PhysicalQAMExpectationConsistentBFM(BinaryDiffusionFlowDecouple):
    """Train with QAM/AWGN/hard-demapped bits, not independent BSC samples."""

    @torch.no_grad()
    def make_physical_observation(self, x_0, path_t, generators=None):
        path_t = path_t.to(device=x_0.device, dtype=torch.long)
        if path_t.shape != (x_0.shape[0],):
            raise ValueError(f"path_t must have shape ({x_0.shape[0]},)")
        snrs = self.snr_values.to(x_0.device)[path_t - 1]
        received = []
        for index in range(x_0.shape[0]):
            generator = None if generators is None else generators[index]
            received.append(
                make_code_noise_single(
                    x_0[index],
                    eb_n0_db=float(snrs[index]),
                    qam_order=self.qam_order,
                    device=x_0.device,
                    generator=generator,
                )
            )
        return torch.stack(received).to(dtype=x_0.dtype), snrs

    def _train_loss(
        self,
        x_0,
        label=None,
        x_t=None,
        path_t=None,
        gt_img=None,
        ae=None,
    ):
        if path_t is None:
            path_t = self.sample_time(x_0.shape[0], x_0.device)
        if x_t is None:
            x_t, physical_snrs = self.make_physical_observation(x_0, path_t)
        else:
            physical_snrs = self.snr_values.to(x_0.device)[path_t - 1]
        stats = super()._train_loss(
            x_0,
            label=label,
            x_t=x_t,
            path_t=path_t,
            gt_img=gt_img,
            ae=ae,
        )
        stats["physical_channel_ber"] = (x_t != x_0).float().mean().detach()
        stats["physical_ebn0_db"] = physical_snrs.mean().detach()
        stats["theoretical_ber"] = (
            self.qam_ber.to(x_0.device)[path_t - 1].mean().detach()
        )
        return stats

    @torch.no_grad()
    def predict_clean_bits_direct(self, received_bits, path_start, temperature=1.0):
        """Threshold the directly supervised X0 logits in one network pass."""
        path_start = int(path_start)
        if path_start == 0:
            return received_bits.float()
        if not 1 <= path_start <= self.num_timesteps:
            raise ValueError(f"path_start must be in [0, {self.num_timesteps}]")
        time_steps = torch.full(
            (received_bits.shape[0],),
            path_start - 1,
            dtype=torch.long,
            device=received_bits.device,
        )
        raw_logits = self._denoise_fn(
            received_bits.float(), time_steps=time_steps
        ) / float(temperature)
        clean_logits = (
            received_bits * (-raw_logits) + (1.0 - received_bits) * raw_logits
            if self.p_flip else raw_logits
        )
        return (clean_logits > 0.0).to(received_bits.dtype)
