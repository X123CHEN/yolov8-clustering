import torch


def set_requires_grad(modules, requires_grad):
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


def _per_sample_lsgan(prediction, is_real):
    """Patch-LSGAN loss reduced spatially, preserving batch shape [B]."""
    target = 1.0 if is_real else 0.0
    return (prediction - target).pow(2).flatten(1).mean(dim=1)


def compute_weighted_generator_adv_loss(student_feat, discriminators, branch_weight, eps=1e-8):
    """Return globally normalized Student adversarial loss and three branch means."""
    weighted_sum = student_feat.sum() * 0.0
    branch_losses = []
    for k, discriminator in enumerate(discriminators):
        per_sample = _per_sample_lsgan(discriminator(student_feat), True)
        weight = branch_weight[:, k]
        weighted_sum = weighted_sum + (per_sample * weight).sum()
        branch_losses.append((per_sample * weight).sum() / weight.sum().clamp_min(eps))
    return weighted_sum / branch_weight.sum().clamp_min(eps), branch_losses


def compute_weighted_discriminator_loss(student_feat, expert_features, discriminators,
                                        branch_weight, eps=1e-8):
    """Return globally normalized real+fake discriminator loss and branch means."""
    weighted_sum = student_feat.sum() * 0.0
    branch_losses = []
    for k, (expert_feat, discriminator) in enumerate(zip(expert_features, discriminators)):
        real = _per_sample_lsgan(discriminator(expert_feat.detach()), True)
        fake = _per_sample_lsgan(discriminator(student_feat.detach()), False)
        per_sample = real + fake
        weight = branch_weight[:, k]
        weighted_sum = weighted_sum + (per_sample * weight).sum()
        branch_losses.append((per_sample * weight).sum() / (2.0 * weight.sum()).clamp_min(eps))
    denominator = (2.0 * branch_weight.sum()).clamp_min(eps)
    return weighted_sum / denominator, branch_losses


def get_adv_warmup_weight(base_weight, epoch, enabled=True, warmup_epochs=10):
    if not enabled or warmup_epochs <= 0:
        return base_weight
    return base_weight * min(1.0, max(0.0, (epoch + 1) / float(warmup_epochs)))

