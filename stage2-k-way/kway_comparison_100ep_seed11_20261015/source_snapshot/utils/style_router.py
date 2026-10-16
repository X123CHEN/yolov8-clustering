from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.style_assignment import compute_soft_assignment


@dataclass
class StyleRoutingOutput:
    full_prob: torch.Tensor
    routing_prob: torch.Tensor
    confidence: torch.Tensor
    final_weight: torch.Tensor
    hard_style: torch.Tensor
    active_mask: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor


def topk_normalize(prob, topk, eps=1e-8):
    """Keep and renormalize the largest topk entries of probabilities [B,K]."""
    if not 1 <= topk <= prob.shape[1]:
        raise ValueError(f"topk must be in [1,{prob.shape[1]}]")
    values, indices = torch.topk(prob, k=topk, dim=1)
    sparse_prob = torch.zeros_like(prob)
    sparse_prob.scatter_(1, indices, values)
    return sparse_prob / sparse_prob.sum(dim=1, keepdim=True).clamp_min(eps)


def build_style_routing(hard_style, distance_sq, mode, topk, temperature,
                        distance_scale, confidence_floor, confidence_power,
                        eps=1e-8):
    """Build detached routing tensors; weights have shape [B,3]."""
    assignment = compute_soft_assignment(
        distance_sq, temperature, distance_scale, eps,
        confidence_floor, confidence_power
    )
    full_prob = assignment.full_prob
    confidence = assignment.confidence
    if mode == "hard":
        routing_prob = F.one_hot(hard_style, num_classes=full_prob.shape[1]).to(full_prob.dtype)
        confidence = torch.ones_like(confidence)
    elif mode == "soft_dense":
        routing_prob = full_prob
        confidence = torch.ones_like(confidence)
    elif mode == "soft_top2":
        routing_prob = topk_normalize(full_prob, topk=topk, eps=eps)
        confidence = torch.ones_like(confidence)
    elif mode == "soft_top2_conf":
        routing_prob = topk_normalize(full_prob, topk=topk, eps=eps)
    else:
        raise ValueError(f"Unsupported style routing mode: {mode}")
    final_weight = routing_prob * confidence.unsqueeze(1)
    return StyleRoutingOutput(
        full_prob.detach(), routing_prob.detach(), confidence.detach(),
        final_weight.detach(), assignment.hard_from_prob.detach(),
        (routing_prob > 0).detach(), assignment.entropy.detach(),
        assignment.normalized_entropy.detach()
    )

