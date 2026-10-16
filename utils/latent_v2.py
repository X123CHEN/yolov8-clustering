"""Pooled adversaries, real GRL, and shared sparse gradient feedback."""
import math
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from nets.backbone import Backbone_pre
from utils.comparison_training import file_hash


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class PooledDiscriminator(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(.2),
            nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, outputs))

    def forward(self, features):
        return self.model(features.mean((2, 3)))


def balanced_domain_ce(logits, q, prior):
    return -(q.detach() / (q.shape[1] * prior.detach()) * F.log_softmax(logits, 1)).sum(1).mean()


class V2Adversary:
    def __init__(self, config, prior, conditional_prior=None):
        self.config = config
        self.prior = torch.as_tensor(prior, device='cuda').detach()
        self.experts, self.discriminators, self.optimizers, self.expert_info = [], [], [], []
        expert_mode = config['architecture'] == 'expert_binary'
        for k in range(config['num_domains'] if expert_mode else 1):
            if expert_mode:
                path = Path(config['expert_paths'][k])
                expert = Backbone_pre(phi='s', pretrained=False)
                state = torch.load(path, map_location='cpu')
                expert.load_state_dict({n[9:]: v for n, v in state.items() if n.startswith('backbone.')}, strict=True)
                self.experts.append(expert.cuda().eval().requires_grad_(False))
                self.expert_info.append(dict(domain=k, path=str(path), sha256=file_hash(path),
                    best=json.loads((path.parent / 'best_checkpoint.json').read_text())))
            # Identical initialization for confusion/GRL; no change to detector RNG.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(config['seed'] + 100 + k)
                d = PooledDiscriminator(1 if expert_mode else config['num_domains']).cuda()
                for layer in d.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
            self.discriminators.append(d)
            self.optimizers.append(torch.optim.Adam(d.parameters(), lr=config['critic_lr'], betas=tuple(config['critic_betas'])))
        self.coefficient = 0.
        self.global_step = 0
        self.last_feedback = {}

    def freeze(self):
        for d in self.discriminators:
            d.eval().requires_grad_(False)

    def raw_generator_loss(self, features, q):
        self.freeze()
        objective = self.config['adversarial_objective']
        if objective == 'expert_lsgan':
            return sum((q[:, k].detach() * (d(features).flatten() - 1).square()).mean()
                       for k, d in enumerate(self.discriminators))
        logits = self.discriminators[0](features)
        if objective == 'grl_domain_ce':
            return balanced_domain_ce(logits, q, self.prior)
        return -F.log_softmax(logits, 1).mean()

    def student_term(self, features, q, raw):
        if self.config['adversarial_objective'] == 'grl_domain_ce':
            # coefficient occurs only inside GRL; D params remain frozen here.
            return balanced_domain_ce(self.discriminators[0](
                GradientReverse.apply(features, self.coefficient)), q, self.prior)
        return self.coefficient * raw

    def feedback(self, detection, raw, features, progress):
        if self.global_step % self.config['feedback_period'] == 0:
            a = torch.autograd.grad(detection, features, retain_graph=True)[0]
            b = torch.autograd.grad(raw, features, retain_graph=True)[0]
            an, bn = float(a.norm()), float(b.norm())
            ratio = self.config['adv_gradient_ratio'] * (2 / (1 + math.exp(-10 * progress)) - 1)
            estimate = min(self.config['adv_lambda_cap'], ratio * an / (bn + self.config['feedback_epsilon']))
            smoothing = self.config['feedback_smoothing']
            self.coefficient = smoothing * self.coefficient + (1 - smoothing) * estimate
            self.last_feedback = dict(step=self.global_step, det_norm=an, raw_adv_norm=bn,
                target_ratio=ratio, coefficient=self.coefficient,
                actual_ratio=self.coefficient * bn / max(an, self.config['feedback_epsilon']),
                cosine=float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)))
        return self.coefficient

    def update(self, images, features, domains, q, conditions=None):
        features, q = features.detach(), q.detach()
        losses, updates = [], []
        if self.config['architecture'] == 'single_kway':
            d, opt = self.discriminators[0], self.optimizers[0]
            d.train().requires_grad_(True); opt.zero_grad(set_to_none=True)
            logits = d(features)
            loss = balanced_domain_ce(logits, q, self.prior)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite domain loss')
            loss.backward(); opt.step()
            p = logits.detach().softmax(1)
            stats = dict(domain_accuracy=float((p.argmax(1) == q.argmax(1)).float().mean()),
                prediction_entropy=float(-(p * p.clamp_min(1e-8).log()).sum(1).mean()),
                predicted_prior=p.mean(0).tolist(), feedback=self.last_feedback)
            value = float(loss.detach()); updates = [1]
        else:
            value = 0.
            for k, (expert, d, opt) in enumerate(zip(self.experts, self.discriminators, self.optimizers)):
                mask = q[:, k] > 0
                if not mask.any(): losses.append(None); updates.append(0); continue
                d.train().requires_grad_(True); opt.zero_grad(set_to_none=True)
                with torch.no_grad(): reference = expert(images[mask])[2]
                per_image = (d(reference).flatten() - 1).square() + d(features[mask]).flatten().square()
                loss = (q[mask, k] * per_image).sum() / (2 * len(images) * self.prior[k])
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite expert loss')
                loss.backward(); opt.step()
                losses.append(float(loss.detach())); updates.append(1)
                value += float(loss.detach()) / len(self.discriminators)
            stats = dict(branch_critic_loss=losses, feedback=self.last_feedback)
        self.global_step += 1
        return value, updates, stats

    def set_prior(self, prior):
        self.prior.copy_(torch.as_tensor(prior, device=self.prior.device))

    def routing_state(self):
        return None

    def restore_routing_state(self, state):
        pass

    def control_state(self):
        return dict(coefficient=self.coefficient, global_step=self.global_step, last_feedback=self.last_feedback)

    def restore_control_state(self, state):
        self.coefficient, self.global_step = state['coefficient'], state['global_step']
        self.last_feedback = state['last_feedback']
