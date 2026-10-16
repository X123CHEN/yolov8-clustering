"""Two expert-reference critics or one K-way classifier; fixed membership targets."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nets.backbone import Backbone_pre
from nets.maddg import Discriminator1
from utils.utils import init_weights
from utils.comparison_training import file_hash
from utils.latent_structure import ConditionalStyleDiscriminator, replace_batch_norm, sampled_branch_risk


def soft_cross_entropy(logits, target):
    return -(target.detach() * F.log_softmax(logits, dim=1)).sum(1).mean()


def adversarial_weight(config, epoch, step=0, steps_per_epoch=1):
    """Linear G-only ramp; D continues learning its source supervision throughout."""
    warmup = config.get("adv_warmup_epochs", 0.0)
    scale = min(1.0, max(0.0, (epoch + step / steps_per_epoch) / warmup)) if warmup else 1.0
    return config["adv_weight"] * scale


@contextmanager
def generator_discriminator(discriminator, bn_mode="running"):
    """Freeze D; optionally use batch BN without changing any running buffers."""
    discriminator.eval().requires_grad_(False)
    batch_norms = [m for m in discriminator.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    previous = [(m, m.training, m.track_running_stats) for m in batch_norms]
    try:
        if bn_mode == "batch":
            for m in batch_norms:
                m.training, m.track_running_stats = True, False
        elif bn_mode != "running":
            raise ValueError(f"Unknown generator BN mode: {bn_mode}")
        yield discriminator
    finally:
        for m, training, tracking in previous:
            m.training, m.track_running_stats = training, tracking


def weighted_generator(discriminators, features, q, bn_mode="running"):
    """Sum branch masses / B: exactly the historical G reduction for one-hot q."""
    value = features.sum() * 0
    for k, discriminator in enumerate(discriminators):
        mask = q[:, k] > 0
        if mask.any():
            with generator_discriminator(discriminator, bn_mode):
                per_image = (discriminator(features[mask]) - 1).square().flatten(1).mean(1)
            value = value + (per_image * q[mask, k].detach()).sum() / len(features)
    return value


class KWayDiscriminator(Discriminator1):
    """Same convolutional trunk as an expert critic; K logits and spatial mean."""
    def __init__(self, count, input_channels=512):
        super().__init__(input_channels=input_channels)
        self.model[-1] = nn.Conv2d(512, count, 4, 1, 1, bias=False)

    def forward(self, features):
        return super().forward(features).mean((2, 3))


class Adversary:
    def __init__(self, config, prior, conditional_prior=None):
        self.config = config
        self.prior = torch.as_tensor(prior, device="cuda").detach()
        self.conditional_prior = (torch.as_tensor(conditional_prior, device='cuda').detach()
                                  if conditional_prior is not None else None)
        self.routing_rng = torch.Generator(device='cpu').manual_seed(config['seed'] + 30000)
        self.experts, self.discriminators, self.optimizers, self.expert_info = [], [], [], []
        expert_mode = config["architecture"] == "expert_binary"
        for k in range(config["num_domains"] if expert_mode else 1):
            if expert_mode:
                path = Path(config["expert_paths"][k])
                expert = Backbone_pre(phi="s", pretrained=False)
                state = torch.load(path, map_location="cpu")
                expert.load_state_dict({name[9:]: value for name, value in state.items() if name.startswith("backbone.")}, strict=True)
                self.experts.append(expert.cuda().eval().requires_grad_(False))
                self.expert_info.append(dict(domain=k, name=f"Cluster_{k}", path=str(path), sha256=file_hash(path),
                                            best=json.loads((path.parent / "best_checkpoint.json").read_text())))
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(config["seed"] + 100 + k)
                if expert_mode:
                    discriminator = Discriminator1(input_channels=512)
                    if config.get('expert_normalization', 'batch') == 'group':
                        discriminator = replace_batch_norm(discriminator, config['expert_group_norm_groups'])
                elif config.get('kway_head', 'legacy_spatial') == 'conditional_style':
                    if self.conditional_prior is None:
                        raise ValueError('Missing source conditional-prior table')
                    discriminator = ConditionalStyleDiscriminator(config['num_domains'], hidden=config['kway_hidden'])
                else:
                    discriminator = KWayDiscriminator(config['num_domains'])
                discriminator = discriminator.cuda()
                init_weights(discriminator, "xavier")
            self.discriminators.append(discriminator)
            self.optimizers.append(torch.optim.Adam(discriminator.parameters(), lr=config["critic_lr"], betas=tuple(config["critic_betas"])))

    def generator_loss(self, features, q, conditions=None):
        bn_mode = self.config.get("generator_bn_mode", "running")
        if self.config["architecture"] == "expert_binary":
            return weighted_generator(self.discriminators, features, q, bn_mode)
        discriminator = self.discriminators[0]
        with generator_discriminator(discriminator, bn_mode):
            if self.config.get('kway_head') == 'conditional_style':
                return soft_cross_entropy(discriminator(features, conditions), self.conditional_prior[conditions.detach()])
            return soft_cross_entropy(discriminator(features), self.prior.expand(len(features), -1))

    def update(self, images, features, domains, q, conditions=None):
        features, q = features.detach(), q.detach()
        updates = [0] * len(self.discriminators)
        if self.config["architecture"] == "single_kway":
            discriminator, optimizer = self.discriminators[0], self.optimizers[0]
            discriminator.train().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            logits = (discriminator(features, conditions) if self.config.get('kway_head') == 'conditional_style'
                      else discriminator(features))
            value = soft_cross_entropy(logits, q)
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite K-way discriminator loss")
            value.backward(); optimizer.step()
            with torch.no_grad():
                p = logits.softmax(1)
                stats = dict(domain_accuracy=float((p.argmax(1) == domains).float().mean()),
                             prediction_entropy=float(-(p * p.clamp_min(1e-8).log()).sum(1).mean()),
                             predicted_prior=p.mean(0).tolist(),
                             target_entropy=float(-(q * q.clamp_min(1e-8).log()).sum(1).mean()))
            return float(value.detach()), [1], stats
        if self.config.get('expert_reference_routing', 'hard') == 'sample_soft':
            return self._sampled_expert_update(images, features, q)
        total, branch_losses = 0., []
        for k, (expert, discriminator, optimizer) in enumerate(zip(self.experts, self.discriminators, self.optimizers)):
            hard_mask = domains == k
            if not hard_mask.any():
                branch_losses.append(None)
                continue  # No real reference for this batch: preserve historical skip behavior.
            discriminator.train().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                reference = expert(images[hard_mask])[2]
            real = (discriminator(reference) - 1).square().mean()
            hard_fake = self.config["assignment"] == "hard" or self.config.get("expert_fake_assignment", "follow") == "hard"
            mask = hard_mask if hard_fake else q[:, k] > 0
            fake_prediction = discriminator(features[mask])
            if hard_fake:
                fake = fake_prediction.square().mean()
            else:
                losses = fake_prediction.square().flatten(1).mean(1)
                fake = (q[mask, k] * losses).sum() / q[mask, k].sum().clamp_min(self.config["membership"]["eps"])
            value = 0.5 * (real + fake)
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite expert discriminator loss")
            value.backward(); optimizer.step()
            total += float(value.detach()) * int(hard_mask.sum()) / len(images)
            updates[k] = 1
            branch_losses.append(float(value.detach()))
        return total, updates, dict(branch_critic_loss=branch_losses)

    def _sampled_expert_update(self, images, features, q):
        # Dedicated CPU RNG never consumes detector/augmentation random streams.
        sampled = torch.multinomial(q.cpu(), 1, generator=self.routing_rng).flatten().to(features.device)
        total, updates, losses, counts = 0., [], [], []
        for k, (expert, d, opt) in enumerate(zip(self.experts, self.discriminators, self.optimizers)):
            mask = sampled == k
            count = int(mask.sum())
            counts.append(count)
            if not count:
                updates.append(0); losses.append(None)
                continue
            d.train().requires_grad_(True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                reference = expert(images[mask])[2]
            value = sampled_branch_risk(d(reference), d(features[mask]), len(images), self.prior[k])
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite sampled expert risk')
            value.backward(); opt.step()
            losses.append(float(value.detach())); updates.append(1)
            total += float(value.detach()) * float(self.prior[k])
        return total, updates, dict(branch_critic_loss=losses, sampled_reference_counts=counts,
                                   reduction='fixed source soft prior; same sampled real/fake branch')

    def routing_state(self):
        return self.routing_rng.get_state()

    def restore_routing_state(self, state):
        if state is None and self.config.get('expert_reference_routing') == 'sample_soft':
            raise ValueError('Missing independent soft-routing RNG checkpoint')
        if state is not None:
            self.routing_rng.set_state(state)
