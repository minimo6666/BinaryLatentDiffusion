"""Expectation-consistent binary flow for a 16QAM communication channel.

This is a received-code denoiser, not an unconditional generator. Training
inputs are hard-demapped codes produced by the physical QAM/AWGN channel and
the clean binary-autoencoder codes are the targets.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.binarylatent_flow_expectation_consistent_retrain import (
    BinaryDiffusionFlowDecouple as _BaseExpectationConsistentBFM,
    focal_loss,
)
from utils.qam_utils import general_m_qam_ber


class BinaryDiffusionFlowDecouple(_BaseExpectationConsistentBFM):
    """QAM communication denoiser with an expectation-consistent bridge."""

    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)
        self.qam_order = int(H.qam_order)
        self.eval_temperature = float(H.temp)
        self.semantic_weight = 1.0
        snr_range = getattr(H, "snr_range", [0, 15])
        self.snr_min = float(snr_range[0])
        self.snr_max = float(snr_range[1])
        if self.snr_max <= self.snr_min:
            raise ValueError(f"invalid snr_range={snr_range}")

        # Path 0 is clean. Paths 1..T use the max-SNR -> min-SNR grid
        # from BinaryDiffusionDSC.
        snr_values = torch.linspace(
            self.snr_max, self.snr_min, self.num_timesteps
        )
        qam_ber = general_m_qam_ber(snr_values, self.qam_order).float()
        tau = (1.0 - 2.0 * qam_ber).clamp(
            min=self.posterior_eps, max=1.0
        )
        self.register_buffer("snr_values", snr_values.float(), persistent=True)
        self.register_buffer("qam_ber", qam_ber, persistent=True)
        self.register_buffer(
            "channel_tau",
            torch.cat([torch.ones(1), tau]).float(),
            persistent=True,
        )

    def sample_time(self, b: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.num_timesteps + 1, (b,), device=device)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Theoretical QAM/BSC marginal used by the reverse bridge."""
        tau_t = self.channel_tau.to(x_0.device)[t.long()]
        tau_t = tau_t.view(-1, *([1] * (x_0.ndim - 1)))
        return (1.0 + (2.0 * x_0.float() - 1.0) * tau_t) / 2.0

    def _cross_step_flip_probability(
        self,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
        reference_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        tau_current = self.channel_tau.to(reference_state_tensor.device)[
            t_current
        ]
        tau_target = self.channel_tau.to(reference_state_tensor.device)[
            t_target
        ]
        gamma = 0.5 * (tau_target - tau_current) / (
            tau_target + self.posterior_eps
        )
        gamma = gamma.clamp(0.0, 0.5)
        return gamma.view(
            -1, *([1] * (reference_state_tensor.ndim - 1))
        )

    def path_step_from_snr(self, snr_db: float) -> int:
        """Map an SNR to the nearest point on the trained QAM path."""
        snr_db = float(snr_db)
        if snr_db > self.snr_max:
            return 0
        if snr_db <= self.snr_min:
            return self.num_timesteps
        return int((self.snr_values - snr_db).abs().argmin().item()) + 1

    def _train_loss(
        self,
        x_0: torch.Tensor,
        label=None,
        x_t: torch.Tensor | None = None,
        path_t: torch.Tensor | None = None,
        gt_img: torch.Tensor | None = None,
        ae=None,
    ):
        """Use the original DSC communication-corruption training path."""
        del label
        x_0 = x_0.float()
        batch_size = x_0.shape[0]
        device = x_0.device
        if path_t is None:
            path_t = self.sample_time(batch_size, device)
        else:
            path_t = path_t.to(device=device, dtype=torch.long)
        if x_t is None:
            x_t = torch.bernoulli(self.q_sample(x_0, path_t))
        else:
            x_t = x_t.to(device=device, dtype=torch.float32)
        if x_0.shape != x_t.shape:
            raise ValueError(
                f"x_0/x_t shape mismatch: {x_0.shape} vs {x_t.shape}"
            )
        if path_t.shape != (x_0.shape[0],):
            raise ValueError(f"path_t must have shape ({x_0.shape[0]},)")
        if (path_t < 1).any() or (path_t > self.num_timesteps).any():
            raise ValueError("training path_t must be in 1..T")

        # Transformer time IDs are 0..T-1, matching BinaryDiffusionDSC.
        network_t = path_t - 1
        raw_logits = self._denoise_fn(x_t, time_steps=network_t)
        if self.p_flip:
            clean_logits = (
                x_t * (-raw_logits) + (1.0 - x_t) * raw_logits
            )
            if self.focal >= 0:
                flip_target = torch.logical_xor(
                    x_0.bool(), x_t.bool()
                ).float()
                bit_loss = focal_loss(
                    raw_logits, flip_target, gamma=self.focal
                )
            else:
                bit_loss = F.binary_cross_entropy_with_logits(
                    clean_logits, x_0, reduction="none"
                )
        else:
            clean_logits = raw_logits
            bit_loss = F.binary_cross_entropy_with_logits(
                clean_logits, x_0, reduction="none"
            )

        if self.loss_final == "weighted":
            weight = (
                1.0 - network_t.float() / self.num_timesteps
            ).view(-1, 1, 1)
        elif self.loss_final == "mean":
            weight = 1.0
        else:
            raise ValueError(f"unknown loss_final={self.loss_final}")

        bce_loss = bit_loss.mean()
        loss = (weight * bit_loss).mean()
        clean_prob = torch.sigmoid(clean_logits)
        semantic_loss = None
        if ae is not None and gt_img is not None:
            soft_code = clean_prob.permute(0, 2, 1).contiguous()
            soft_code = soft_code.reshape(
                batch_size,
                self.codebook_size,
                self.shape[1],
                self.shape[2],
            )
            pred_img, _, _ = ae(None, code=soft_code)
            semantic_loss = F.mse_loss(pred_img, gt_img)
            loss = loss + self.semantic_weight * semantic_loss
        stats = {
            "loss": loss,
            "bce_loss": bce_loss,
            "acc": ((clean_prob > 0.5) == x_0.bool()).float().mean(),
            "channel_ber": (x_t != x_0).float().mean(),
            "channel_snr": self.snr_values[network_t].mean(),
        }
        if semantic_loss is not None:
            stats["semantic_loss"] = semantic_loss.detach()

        if self.aux > 0:
            t_target = self._sample_aux_target_time(path_t)
            reverse_prob = (
                self._expectation_consistent_reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=path_t,
                    t_target=t_target,
                )
            )
            bridge_zero, bridge_one = (
                self._endpoint_bridge_probabilities(
                    x_t=x_t,
                    t_current=path_t,
                    t_target=t_target,
                )
            )
            reverse_target = (
                (1.0 - x_0) * bridge_zero + x_0 * bridge_one
            )
            with torch.autocast(
                device_type=x_t.device.type, enabled=False
            ):
                aux_loss = F.binary_cross_entropy(
                    reverse_prob.float().clamp(1e-6, 1.0 - 1e-6),
                    reverse_target.float().clamp(0.0, 1.0),
                    reduction="none",
                )
            aux_loss = (weight * aux_loss).mean()
            loss = loss + self.aux * aux_loss
            stats.update(
                {
                    "loss": loss,
                    "aux loss": aux_loss,
                    "reverse_mae": (
                        reverse_prob - reverse_target
                    ).abs().mean(),
                    "mean_aux_gap": (
                        path_t - t_target
                    ).float().mean(),
                }
            )
        return stats

    def forward(
        self,
        x,
        label=None,
        x_t=None,
        path_t=None,
        gt_img=None,
        ae=None,
        **unused,
    ):
        del unused
        return self._train_loss(
            x,
            label=label,
            x_t=x_t,
            path_t=path_t,
            gt_img=gt_img,
            ae=ae,
        )

    @torch.no_grad()
    def sample_from_channel(
        self,
        x_t: torch.Tensor,
        path_start: int,
        *,
        temp: float = 0.9,
        sample_steps: int | None = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        """Denoise received bits; random initialization is never used."""
        if temp <= 0:
            raise ValueError(f"temp must be positive, got {temp}")
        path_start = int(path_start)
        if path_start < 0 or path_start > self.num_timesteps:
            raise ValueError(f"path_start must be in [0,T], got {path_start}")
        x_t = x_t.float()
        if path_start == 0:
            return x_t

        if sample_steps is None:
            sample_steps = path_start
        sample_steps = max(1, min(int(sample_steps), path_start))
        reverse_path = np.arange(1, path_start + 1)
        if sample_steps != path_start:
            indices = np.linspace(
                0, path_start - 1, sample_steps
            ).astype(np.int64)
            reverse_path = reverse_path[indices]
        reverse_path = reverse_path[::-1]

        batch_size = x_t.shape[0]
        trajectory = [x_t] if return_all else None
        for index, current_value in enumerate(reverse_path):
            t_current = torch.full(
                (batch_size,),
                int(current_value),
                device=x_t.device,
                dtype=torch.long,
            )
            raw_logits = self._denoise_fn(
                x_t, time_steps=t_current - 1
            ) / temp
            clean_logits = (
                x_t * (-raw_logits) + (1.0 - x_t) * raw_logits
                if self.p_flip
                else raw_logits
            )
            clean_prob = torch.sigmoid(clean_logits)
            target_value = (
                0
                if index == len(reverse_path) - 1
                else int(reverse_path[index + 1])
            )
            if target_value == 0:
                x_t = (
                    (clean_prob > 0.5).float()
                    if self.hard_final
                    else torch.bernoulli(clean_prob)
                )
            else:
                t_target = torch.full_like(t_current, target_value)
                reverse_prob = self._reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=t_current,
                    t_target=t_target,
                )
                x_t = torch.bernoulli(reverse_prob)
            if return_all:
                trajectory.append(x_t)
        return torch.stack(trajectory) if return_all else x_t


def qam_path_step_from_snr(
    snr_db: float,
    snr_min: float = 0.0,
    snr_max: float = 15.0,
    total_steps: int = 64,
) -> int:
    """Pure helper mirroring path_step_from_snr."""
    if snr_db > snr_max:
        return 0
    if snr_db <= snr_min:
        return int(total_steps)
    values = np.linspace(snr_max, snr_min, int(total_steps))
    return int(np.abs(values - float(snr_db)).argmin()) + 1
