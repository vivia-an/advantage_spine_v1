"""Loss-side negative-advantage weighting used by the Spine recipe."""

from __future__ import annotations

import torch
from torch import Tensor


def apply_negative_advantage_weight(
    pg_losses: Tensor,
    advantages: Tensor,
    neg_loss_coef: float = 0.1,
) -> Tensor:
    """Scale clipped policy-loss terms with negative standardized advantage.

    Call this after PPO ratio and dual clipping and before response-mask loss
    aggregation. Add the KL loss separately after aggregating the returned
    policy-gradient loss so that ``neg_loss_coef`` does not scale KL.
    """
    if not 0.0 <= neg_loss_coef <= 1.0:
        raise ValueError("neg_loss_coef must be in [0, 1]")
    neg_scale = torch.where(
        advantages < 0,
        torch.as_tensor(
            neg_loss_coef, device=pg_losses.device, dtype=pg_losses.dtype
        ),
        torch.ones_like(pg_losses),
    )
    return pg_losses * neg_scale
