"""Single hard-domain classifier on the historical Student feature tap."""
import math

import torch
from torch.nn import functional as F

from utils.latent_adversarial import KWayDiscriminator


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, coefficient):
        ctx.coefficient = float(coefficient)
        return feature.view_as(feature)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def dann_coefficient(step, total_steps):
    progress = min(1., max(0., step / max(total_steps, 1)))
    return 2. / (1. + math.exp(-10. * progress)) - 1.


def hard_domain_loss(discriminator, feature, domains, coefficient, valid_mask=None):
    logits = discriminator(GradientReverse.apply(feature, coefficient))
    if valid_mask is not None:
        logits, domains = logits[valid_mask], domains[valid_mask]
    loss = F.cross_entropy(logits, domains) if len(domains) else logits.sum() * 0.
    return loss, logits


def confusion_counts(logits, domains):
    k = logits.shape[1]
    return torch.bincount(domains * k + logits.detach().argmax(1), minlength=k * k).reshape(k, k)
