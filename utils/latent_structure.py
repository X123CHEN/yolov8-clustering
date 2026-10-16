"""Sample-independent auxiliary heads and source-only conditional metadata.

Legacy adversaries do not use these components. No target or validation labels
enter a prior, and conditions describe the original annotated training image.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def source_conditions(records, owners, classes, domains, smoothing):
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError('Conditional-prior smoothing must be finite and positive')
    conditions = []
    counts = np.zeros((2 ** classes, domains), dtype=np.int64)
    for key in sorted(records):
        mask = 0
        for field in records[key].split()[1:]:
            x1, y1, x2, y2, label = map(int, field.split(','))
            if not 0 <= label < classes:
                raise ValueError(f'Invalid source class: {field}')
            if x2 > x1 and y2 > y1:
                mask |= 1 << label
        conditions.append(mask)
        counts[mask, owners[key]] += 1
    global_prior = counts.sum(0) / counts.sum()
    table = (counts + smoothing * global_prior) / (counts.sum(1, keepdims=True) + smoothing)
    return np.asarray(conditions, dtype=np.int64), table.astype(np.float32), dict(
        definition='original source image valid-box class-presence bitmask',
        units='one image, not instances', counts=counts.tolist(),
        global_hard_prior=global_prior.tolist(), smoothing=float(smoothing),
        conditional_hard_prior=table.tolist(), images=len(conditions))


def style_statistics(features, eps=1e-8):
    flattened = features.flatten(2)
    return torch.cat((flattened.mean(2), (flattened.var(2, unbiased=True) + eps).sqrt()), dim=1)


class ConditionalStyleDiscriminator(nn.Module):
    """One K-way head; multiplicative style/condition interactions plus offsets."""
    def __init__(self, domains, channels=128, conditions=16, hidden=128):
        super().__init__()
        self.conditions = conditions
        self.style = nn.Sequential(nn.Linear(2 * channels, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(.2))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * conditions + conditions, hidden), nn.LayerNorm(hidden),
            nn.LeakyReLU(.2), nn.Linear(hidden, domains))

    def forward(self, features, conditions):
        c = F.one_hot(conditions.detach(), self.conditions).to(features.dtype)
        s = self.style(style_statistics(features))
        joint = (s.unsqueeze(2) * c.unsqueeze(1)).flatten(1)
        return self.classifier(torch.cat((joint, c), dim=1))


def replace_batch_norm(module, groups=32):
    """Keep the expert CNN geometry, remove dependence on batch populations."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            if child.num_features % groups:
                raise ValueError('Group count must divide every critic channel count')
            setattr(module, name, nn.GroupNorm(groups, child.num_features, eps=child.eps))
        else:
            replace_batch_norm(child, groups)
    return module


def sampled_branch_risk(real_prediction, fake_prediction, batch_size, prior):
    """Indicator sampling uses a FIXED denominator, never random group size."""
    per_image = ((real_prediction - 1).square().flatten(1).mean(1)
                 + fake_prediction.square().flatten(1).mean(1))
    return per_image.sum() / (2 * batch_size * prior.detach())
