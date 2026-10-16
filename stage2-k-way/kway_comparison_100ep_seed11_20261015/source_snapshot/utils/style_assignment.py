from dataclasses import dataclass
import math

import torch


@dataclass
class StyleAssignmentOutput:
    full_prob: torch.Tensor
    hard_from_prob: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    confidence: torch.Tensor


def compute_soft_assignment(distance_sq, temperature, distance_scale, eps=1e-8,
                            confidence_floor=0.2, confidence_power=1.0):
    """Convert squared distances [B,K] to probabilities and confidence [B]."""
    if distance_sq.ndim != 2:
        raise ValueError(f"distance_sq must be [B,K], got {tuple(distance_sq.shape)}")
    if temperature <= 0 or distance_scale <= 0:
        raise ValueError("temperature and distance_scale must be positive")
    if not 0.0 <= confidence_floor <= 1.0 or confidence_power <= 0:
        raise ValueError("invalid confidence parameters")
    scaled_logits = -distance_sq / (temperature * distance_scale + eps)
    full_prob = torch.softmax(scaled_logits, dim=1)
    hard_from_prob = torch.argmax(full_prob, dim=1)
    entropy = -(full_prob * torch.log(full_prob.clamp_min(eps))).sum(dim=1)
    normalized_entropy = entropy / math.log(full_prob.shape[1])
    raw_confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)
    confidence = (confidence_floor + (1.0 - confidence_floor) * raw_confidence)
    confidence = confidence.pow(confidence_power)
    return StyleAssignmentOutput(full_prob, hard_from_prob, entropy,
                                 normalized_entropy, confidence)

