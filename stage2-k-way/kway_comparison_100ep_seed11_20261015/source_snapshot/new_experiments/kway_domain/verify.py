"""Real source-batch checks before the four 60-epoch training runs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nets.yolo import YoloBody
from nets.yolo_training import Loss
from nets.maddg import Discriminator1
from utils.utils import seed_everything, init_weights, get_classes
from utils.comparison_training import (ROOT, prepare_data, DomainDataset, domain_collate,
    domain_triplet, atomic_json)
from new_experiments.kway_domain.discriminator import (GradientReverse, KWayDiscriminator,
    hard_domain_loss, dann_coefficient)
from new_experiments.kway_domain.cluster_index import audit_source_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = 'cuda'
    result = dict(seed=args.seed, epochs=args.epochs, labels=audit_source_labels(), checks={}, calibrations={})
    # GRL affects feature gradients only, for general K and B=1.
    for k in (2, 3):
        head = nn.Linear(5, k).to(device)
        z = torch.randn(4, 5, device=device, requires_grad=True)
        ids = torch.arange(4, device=device) % k
        plain = F.cross_entropy(head(z), ids)
        reverse = F.cross_entropy(head(GradientReverse.apply(z, .37)), ids)
        ordinary = torch.autograd.grad(plain, (z, *head.parameters()), retain_graph=True)
        reversed_ = torch.autograd.grad(reverse, (z, *head.parameters()))
        torch.testing.assert_close(reversed_[0], -.37 * ordinary[0])
        for a, b in zip(ordinary[1:], reversed_[1:]): torch.testing.assert_close(a, b)
        d = KWayDiscriminator(k).cuda().train()
        assert d(torch.randn(1, 512, 20, 20, device=device)).shape == (1, k)
        assert d(torch.randn(12, 512, 20, 20, device=device)).shape == (12, k)
        del d
    result['checks']['grl_sign_head_gradients_b1_b12'] = True
    result['checks']['schedule_zero_and_progress'] = dann_coefficient(0, 100) == 0 and dann_coefficient(99, 100) > .99
    _, classes = get_classes(str(ROOT / 'model_data/rdd_classes.txt'))
    seed_everything(args.seed)
    model = YoloBody([640, 640], classes, 's', pretrained=False).cuda()
    state = torch.load(ROOT / 'model_data/yolov8_s.pth', map_location='cpu')
    missing, _ = model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
    if any(k.startswith('backbone.') for k in missing):
        raise RuntimeError('Incomplete initial Student backbone')
    initial_student = deepcopy(model.state_dict())
    loss_fn = Loss(model)
    for kind in ('country', 'latent'):
        names, _, _, rows, records, owners, _, _ = prepare_data(kind + '_unique')
        dataset = DomainDataset(rows, records, owners, classes, args.epochs)
        indices = []
        for domain in range(len(names)):
            indices.extend([i for i, c in enumerate(dataset.domains) if c == domain][:12])
        # Three source-only mini-batches, each including all supervised domains.
        batches = [[indices[d * 12 + j] for d in range(len(names))
                    for j in range(offset, offset + 12 // len(names))]
                   for offset in range(0, 12, 12 // len(names))][:3]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(args.seed + 100)
            domain_head = KWayDiscriminator(len(names)).cuda()
            init_weights(domain_head, 'xavier')
        old_heads = []
        for domain in range(len(names)):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(args.seed + 100 + domain)
                old = Discriminator1(input_channels=512).cuda()
                init_weights(old, 'xavier')
            old_heads.append(old.eval().requires_grad_(False))
        norm_records, probe_features, probe_ids = [], [], []
        for index, selected in enumerate(batches):
            seed_everything(args.seed + index)
            model.load_state_dict(initial_student); model.train()
            images, targets, ids = domain_collate([dataset[i] for i in selected])
            images, targets, ids = images.cuda(), targets.cuda(), ids.cuda()
            outputs = model(images)
            assert outputs[5].shape == (12, 512, 20, 20)
            detection = loss_fn(outputs[:5], targets)
            fab, anchors = domain_triplet(outputs, targets, ids, len(names))
            domain_head.train()
            ce, logits = hard_domain_loss(domain_head, outputs[5], ids, 1.)
            legacy = outputs[5].sum() * 0
            for d, old in enumerate(old_heads):
                mask = ids == d
                legacy += (old(outputs[5][mask]) - 1).square().flatten(1).mean(1).sum() / len(ids)
            norms = {}
            for key, value in [('detection', detection), ('fab', .3 * fab), ('legacy_raw', legacy), ('kway_raw', ce)]:
                grad = torch.autograd.grad(value, outputs[5], retain_graph=True)[0]
                norms[key] = float(grad.norm())
            backbone_gradient = torch.autograd.grad(ce, tuple(model.backbone.parameters()), retain_graph=True, allow_unused=True)
            backbone_norm = float(torch.stack([g.square().sum() for g in backbone_gradient if g is not None]).sum().sqrt())
            if norms['fab'] <= 0 or backbone_norm <= 0 or not torch.isfinite(ce):
                raise RuntimeError('FAB or Student domain gradient is inactive')
            # Domain weight zero produces exactly the shared stage-one task graph.
            reference = detection + .3 * fab
            disabled = detection + .3 * fab + 0. * ce
            torch.testing.assert_close(reference, disabled, rtol=0, atol=0)
            a = torch.autograd.grad(reference, outputs[5], retain_graph=True)[0]
            b = torch.autograd.grad(disabled, outputs[5], retain_graph=True)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            norms.update(batch=index, fab_loss=float(fab), anchors=anchors, student_domain_gradient=backbone_norm,
                         raw_gradient_ratio=norms['kway_raw'] / norms['legacy_raw'])
            norm_records.append(norms)
            probe_features.append(outputs[5].detach()); probe_ids.append(ids)
            del outputs, backbone_gradient, reference, disabled, a, b, detection, fab, ce, legacy, logits, grad
        # Fixed coefficient, calibrated once using only the same source mini-batches.
        alpha = float(np.median([.7 * row['legacy_raw'] / row['kway_raw'] for row in norm_records]))
        result['calibrations'][kind] = dict(alpha_domain=alpha, historical_effective_weight=.7,
            formula='median(0.7 * legacy_raw_feature_grad_norm / kway_raw_feature_grad_norm)',
            batches=norm_records, scope='fixed source training batches; r=1; no target data')
        # A frozen Student sample must support domain discrimination without GRL.
        features, labels = torch.cat(probe_features), torch.cat(probe_ids)
        probe = deepcopy(domain_head).train()
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        with torch.no_grad(): before = float(F.cross_entropy(probe(features), labels))
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            probe_loss = F.cross_entropy(probe(features), labels)
            probe_loss.backward(); opt.step()
        with torch.no_grad():
            prediction = probe(features).argmax(1)
            recall = [float((prediction[labels == k] == k).float().mean()) for k in range(len(names))]
            balanced_accuracy = float(np.mean(recall))
        if balanced_accuracy <= 1 / len(names) + .15 or float(probe_loss) >= before:
            raise RuntimeError('Frozen source-sample domain probe cannot learn fixed labels')
        result['checks'][kind] = dict(real_batch_shapes=True, student_gradient=True, fab_gradient=True,
            disabled_domain_task_parity=True, frozen_sample_probe_accuracy=balanced_accuracy,
            probe_initial_ce=before, probe_final_ce=float(probe_loss),
            probe_scope='source training samples only; connectivity/learnability, not held-out domain invariance')
        del probe, opt, features, domain_head, old_heads, probe_features
        torch.cuda.empty_cache()
    result['passed'] = True
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
