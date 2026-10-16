"""Controlled country/K=2 comparison, invoked via train_style_adv.py --experiment.

Repeated experiments preserve paired TXT rows; unique experiments traverse the
same canonical image order and carry different hard domain IDs. No style cache.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from functools import partial
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nets.yolo import YoloBody
from nets.backbone import Backbone_pre
from nets.maddg import Discriminator1
from nets.yolo_training import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory, EvalCallback, append_metrics_epoch
from utils.dataloader import YoloDataset_eval, yolo_dataset_collate_eval
from utils.utils import get_classes, seed_everything, worker_init_fn, init_weights

ROOT = Path(__file__).resolve().parents[1]
EXPERT_ROOT = ROOT.parent / 'yolov8-pretrain/results/expert_pretrain_20261004_200ep'
EXPERIMENTS = {
    'baseline': ('country', False, False),
    'country_repeat': ('country', False, True),
    'latent_repeat': ('latent', False, True),
    'country_unique': ('country', True, True),
    'latent_unique': ('latent', True, True),
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def read_records(path, domains):
    rows, records, owners = [], {}, {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split('&')
        if len(parts) != domains:
            raise ValueError(f'{path}: expected {domains} domains, got {len(parts)}')
        row = []
        for domain, part in enumerate(parts):
            fields = part.split()
            image = str(Path(fields[0]).resolve(strict=True))
            record = ' '.join([image] + fields[1:])
            if image in records and (records[image] != record or owners[image] != domain):
                raise ValueError(f'Conflicting annotation/domain for {image}')
            records[image], owners[image] = record, domain
            row.append(image)
        rows.append(row)
    return rows, records, owners


def prepare_data(experiment):
    kind, unique, adversarial = EXPERIMENTS[experiment]
    names = ['Japan', 'India', 'United_States'] if kind == 'country' else ['Cluster_0', 'Cluster_1']
    base = ROOT / ('datasets/RDD2022/ImageSets' if kind == 'country' else 'latent_domain_analysis/domain_splits')
    train = base / ('251030_train.txt' if kind == 'country' else 'combined_train.txt')
    val = base / ('251030_val.txt' if kind == 'country' else 'combined_val.txt')
    rows, records, owners = read_records(train, len(names))
    _, validation, _ = read_records(val, 1)
    if records.keys() & validation.keys():
        raise ValueError('Training/validation overlap')
    keys = sorted(records)
    if unique:
        rows = [[key] for key in keys]
    val_lines = [validation[key] for key in sorted(validation)]
    return names, train, val, rows, records, owners, val_lines, adversarial


class DomainDataset(Dataset):
    def __init__(self, rows, records, owners, classes, epochs):
        keys = sorted(records)
        indices = {key: i for i, key in enumerate(keys)}
        self.rows = [[indices[key] for key in row] for row in rows]
        self.domains = [owners[key] for key in keys]
        self.base = YoloDataset_eval([records[key] for key in keys], [640, 640], classes,
                                    epoch_length=epochs, mosaic=False, mixup=False,
                                    mosaic_prob=0, mixup_prob=0, train=True, special_aug_ratio=0)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return [(self.base[item], self.domains[item]) for item in self.rows[index]]


def domain_collate(batch):
    samples = [item for row in batch for item in row]
    images, targets = yolo_dataset_collate_eval([item[0] for item in samples])
    return images, targets, torch.tensor([item[1] for item in samples], dtype=torch.long)


def pooled_objects(feature, targets):
    """Mean-pool each box in [y,x] order, clamp to the actual feature size.

    Keep bbox-to-feature extraction differentiable, including degenerate boxes.
    """
    height, width = feature.shape[-2:]
    embeddings = []
    for row in targets.detach().cpu().tolist():
        image, _, cx, cy, bw, bh = row
        x1 = min(width - 1, max(0, math.floor((cx - bw / 2) * width)))
        y1 = min(height - 1, max(0, math.floor((cy - bh / 2) * height)))
        x2 = min(width, max(x1 + 1, math.ceil((cx + bw / 2) * width)))
        y2 = min(height, max(y1 + 1, math.ceil((cy + bh / 2) * height)))
        embeddings.append(feature[int(image), :, y1:y2, x1:x2].mean((1, 2)))
    return torch.stack(embeddings) if embeddings else feature.new_empty((0, feature.shape[1]))


def mined_triplet(features, labels, margin):
    """Class-based batch-hard mining; exclude self and invalid anchors."""
    if len(features) < 3:
        return features.sum() * 0, 0
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return features.sum() * 0, 0
    # Match the original Euclidean metric with a nonzero sqrt floor.
    distance = (features.square().sum(1)[:, None] + features.square().sum(1)[None, :]
                - 2 * features @ features.t()).clamp_min(1e-12).sqrt()
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values[valid]
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values[valid]
    return (hardest_positive - hardest_negative + margin).clamp_min(0).mean(), int(valid.sum())


def domain_triplet(outputs, targets, domains, count):
    zero = outputs[5].sum() * 0
    if not len(targets):
        return zero, 0
    labels = targets[:, 1].long()
    object_domains = domains[targets[:, 0].long()]
    # Pair pools plus the global pool for K>2; K=2's identical global pool is counted once.
    pools = list(itertools.combinations(range(count), 2))
    if count > 2:
        pools.append(tuple(range(count)))
    scale_losses, anchors = [], 0
    for feature in (outputs[5], outputs[6]):
        embeddings = pooled_objects(feature, targets)
        inter, intra = [], []
        for pool in pools:
            if any(not (object_domains == domain).any() for domain in pool):
                continue
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for domain in pool:
                mask |= object_domains == domain
            value, n = mined_triplet(embeddings[mask], labels[mask], 0.5)
            if n:
                inter.append(value)
                anchors += n
        for domain in range(count):
            mask = object_domains == domain
            value, n = mined_triplet(embeddings[mask], labels[mask], 0.1)
            if n:
                intra.append(value)
                anchors += n
        scale_losses.append((torch.stack(inter).mean() if inter else zero)
                            + 0.1 * (torch.stack(intra).mean() if intra else zero))
    return torch.stack(scale_losses).mean(), anchors


def require_finite(value, description):
    if not torch.isfinite(value).all():
        raise RuntimeError(f'Non-finite {description}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--results_path', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=12, help='Total student images per batch, not per domain')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--smoke_steps', type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size < 2 or args.batch_size % 6 or args.num_workers < 0 or args.smoke_steps < 0:
        parser.error('Use positive epochs, a total batch size divisible by 6, and nonnegative workers/smoke_steps')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    save = args.results_path.resolve() / args.experiment
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'run_config.json').exists():
        raise FileExistsError(f'Use a fresh results_path; {save} already has a run')
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    names, train_path, val_path, rows, records, owners, val_lines, adv = prepare_data(args.experiment)
    group = len(rows[0])
    class_names, classes = get_classes(str(ROOT / 'model_data/rdd_classes.txt'))
    dataset = DomainDataset(rows, records, owners, classes, args.epochs)
    if args.smoke_steps:
        val_lines = val_lines[:args.batch_size + 1]
    validation = YoloDataset_eval(val_lines, [640, 640], classes, epoch_length=args.epochs,
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                  train=False, special_aug_ratio=0)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size // group, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=domain_collate,
                        worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed), generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=yolo_dataset_collate_eval)
    model = YoloBody([640, 640], classes, 's', pretrained=False).cuda()
    initial_path = ROOT / 'model_data/yolov8_s.pth'
    initial = torch.load(initial_path, map_location='cpu')
    compatible = {k: v for k, v in initial.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    missing, _ = model.load_state_dict(compatible, strict=False)
    if any(k.startswith('backbone.') for k in missing):
        raise RuntimeError(f'Incomplete initial backbone: {missing}')
    # Capture before allocating domain-dependent modules: all five students must match.
    student_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    loss_fn = Loss(model)
    ema = ModelEMA(model)
    experts, discriminators, critic_opts, expert_info = [], [], [], []
    if adv:
        for domain, name in enumerate(names):
            path = EXPERT_ROOT / name / 'best_epoch_weights.pth'
            expert = Backbone_pre(phi='s', pretrained=False)
            checkpoint = torch.load(path, map_location='cpu')
            expert.load_state_dict({k[9:]: v for k, v in checkpoint.items() if k.startswith('backbone.')}, strict=True)
            experts.append(expert.cuda().eval().requires_grad_(False))
            # Corresponding branch indices initialize identically in all DG experiments.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(args.seed + 100 + domain)
                discriminator = Discriminator1(input_channels=512).cuda()
                init_weights(discriminator, 'xavier')
            discriminators.append(discriminator)
            critic_opts.append(torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999)))
            expert_info.append(dict(domain=domain, name=name, path=str(path), sha256=file_hash(path),
                                    best=json.loads((path.parent / 'best_checkpoint.json').read_text())))
    pg0, pg1, pg2 = [], [], []
    for _, module in model.named_modules():
        if isinstance(getattr(module, 'bias', None), nn.Parameter):
            pg2.append(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            pg0.append(module.weight)
        elif isinstance(getattr(module, 'weight', None), nn.Parameter):
            pg1.append(module.weight)
    lr = min(max(args.batch_size / 64 * 0.01, 5e-4), 0.05)
    min_lr = min(max(args.batch_size / 64 * 0.0001, 5e-6), 0.0005)
    optimizer = torch.optim.SGD(pg0, lr, momentum=0.937, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})
    optimizer.add_param_group({'params': pg2})
    schedule = get_lr_scheduler('cos', lr, min_lr, args.epochs)
    config = dict(experiment=args.experiment, epochs=args.epochs, seed=args.seed,
                  num_workers=args.num_workers,
                  total_batch_size=args.batch_size, paired_domains_per_row=group,
                  train_rows=len(rows), train_appearances=sum(map(len, rows)), unique_train_images=len(records),
                  train_order_sha256=hashlib.sha256('\n'.join('|'.join(row) for row in rows).encode()).hexdigest(),
                  validation_images=len(val_lines), steps_per_epoch=len(loader),
                  train_txt=str(train_path), val_txt=str(val_path), train_sha256=file_hash(train_path), val_sha256=file_hash(val_path),
                  validation_order_sha256=hashlib.sha256('\n'.join(val_lines).encode()).hexdigest(),
                  initial_weights=str(initial_path), initial_sha256=file_hash(initial_path), student_initial_sha256=student_hash,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), domains=names,
                  experts=expert_info, discriminators=len(discriminators), optimizer='SGD', momentum=0.937,
                  weight_decay=5e-4, nominal_lr=0.01, fitted_lr=lr, min_lr=min_lr, scheduler='cos',
                  critic_optimizer='Adam', critic_lr=1e-5, critic_betas=[0.9, 0.999],
                  input_shape=[640, 640], phi='s', fp16=False, freeze_train=False,
                  mosaic=False, mixup=False, fft=False, mixstyle=False, routing='hard_domain_id',
                  detection_weight=1.0, adversarial_weight=0.7 if adv else 0.0, triplet_weight=0.3 if adv else 0.0,
                  adversarial_feature='feat3_20x20', triplet_features=['feat3_20x20', 'feat2_40x40'],
                  triplet_inter_margin=0.5, triplet_intra_margin=0.1, triplet_intra_weight=0.1,
                  triplet_reduction='mean over valid pools; mean over scales; no duplicated K=2 global pool',
                  triplet_mining='class-based batch-hard in domain pools; exclude self/invalid anchors',
                  val_loss_reduction='image-weighted batch mean', best_criterion='minimum val_loss',
                  smoke_steps=args.smoke_steps, cluster1_expert_training='stopped during epoch 166; best epoch 52 authorized')
    atomic_json(save / 'run_config.json', config)
    print('RUN_CONFIG ' + json.dumps(config, ensure_ascii=False), flush=True)
    history = LossHistory(str(save / 'metrics'), model, [640, 640])
    evaluator = EvalCallback(model, [640, 640], class_names, classes, val_lines, str(save / 'metrics'), True,
                             map_out_path=str(save / 'temp_map_out'), eval_flag=not args.smoke_steps, period=1)
    best = float('inf')
    epochs = 1 if args.smoke_steps else args.epochs
    for epoch in range(epochs):
        started = time.time()
        generator.manual_seed(args.seed + epoch)
        dataset.base.epoch_now = epoch
        model.train()
        set_optimizer_lr(optimizer, schedule, epoch)
        torch.cuda.reset_peak_memory_stats()
        totals = dict(total=0., detection=0., adversarial=0., triplet=0., critic=0.)
        images_seen, steps, anchors = 0, 0, 0
        domain_images = [0] * len(names)
        discriminator_updates = [0] * len(names)
        effective_steps = min(len(loader), args.smoke_steps) if args.smoke_steps else len(loader)
        for images, targets, domains in tqdm(itertools.islice(loader, effective_steps), total=effective_steps,
                                             desc=f'{args.experiment} epoch {epoch + 1}/{args.epochs}', mininterval=10):
            images, targets, domains = images.cuda(non_blocking=True), targets.cuda(non_blocking=True), domains.cuda()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            detection = loss_fn(outputs[:5], targets)
            adversarial = outputs[5].sum() * 0
            triplet, valid_anchors = adversarial, 0
            n = len(images)
            for domain in range(len(names)):
                domain_images[domain] += int((domains == domain).sum())
            if adv:
                for domain, discriminator in enumerate(discriminators):
                    discriminator.eval().requires_grad_(False)
                    mask = domains == domain
                    if mask.any():
                        prediction = discriminator(outputs[5][mask])
                        adversarial = adversarial + (prediction - 1).square().flatten(1).mean(1).sum() / n
                triplet, valid_anchors = domain_triplet(outputs, targets, domains, len(names))
            total = detection + 0.7 * adversarial + 0.3 * triplet
            require_finite(total, 'student loss')
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            critic = 0.
            if adv:
                for domain, (expert, discriminator, opt) in enumerate(zip(experts, discriminators, critic_opts)):
                    mask = domains == domain
                    if not mask.any():
                        continue
                    discriminator.train().requires_grad_(True)
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        reference = expert(images[mask])[2]
                    value = 0.5 * ((discriminator(reference) - 1).square().mean()
                                   + discriminator(outputs[5][mask].detach()).square().mean())
                    require_finite(value, 'discriminator loss')
                    value.backward()
                    opt.step()
                    critic += float(value.detach()) * int(mask.sum()) / n
                    discriminator_updates[domain] += 1
            for key, value in [('total', total), ('detection', detection), ('adversarial', adversarial), ('triplet', triplet)]:
                totals[key] += float(value.detach()) * n
            totals['critic'] += critic * n
            anchors += valid_anchors
            images_seen += n
            steps += 1
            if steps == 1 or steps % 100 == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=steps, detection=float(detection.detach()),
                                      adversarial=float(adversarial.detach()), triplet=float(triplet.detach()),
                                      critic=critic, valid_triplet_anchors=valid_anchors, gradient_norm=float(norm))), flush=True)
        train_time = time.time() - started
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        if not args.smoke_steps and images_seen != sum(map(len, rows)):
            raise RuntimeError('Incomplete training traversal')
        val_started = time.time()
        val_total, val_count = 0., 0
        ema.ema.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                value = loss_fn(ema.ema(images)[:5], targets)
                require_finite(value, 'validation loss')
                val_total += float(value) * len(images)
                val_count += len(images)
        if val_count != len(validation):
            raise RuntimeError('Incomplete validation traversal')
        val_loss = val_total / val_count
        val_time = time.time() - val_started
        if val_loss < best:
            best = val_loss
            atomic_save(ema.ema.state_dict(), save / 'best_epoch_weights.pth')
            atomic_json(save / 'best_checkpoint.json', dict(epoch=epoch + 1, val_loss=val_loss,
                                                          criterion='minimum val_loss', validation_images=val_count))
        atomic_save(ema.ema.state_dict(), save / 'last_epoch_weights.pth')
        # Keep optimizer/EMA/discriminator state so interrupted runs can be resumed faithfully.
        atomic_save(dict(epoch=epoch + 1, model=model.state_dict(), ema=ema.ema.state_dict(), ema_updates=ema.updates,
                         optimizer=optimizer.state_dict(), discriminators=[d.state_dict() for d in discriminators],
                         critic_optimizers=[o.state_dict() for o in critic_opts], best_val_loss=best, config=config,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         numpy_rng=np.random.get_state(), python_rng=random.getstate()), save / 'training_state.pt')
        history.append_loss(epoch + 1, totals['total'] / images_seen, val_loss)
        map50 = evaluator.on_epoch_end(epoch + 1, ema.ema)
        append_metrics_epoch(str(save / 'metrics'), dict(epoch=epoch + 1, train_loss=totals['total'] / images_seen,
                            val_loss=val_loss, map_50=map50, train_time_s=train_time, val_time_s=val_time,
                            epoch_time_s=time.time() - started, steps=steps, images_per_step=args.batch_size,
                            train_images_per_s=images_seen / train_time, loss_yolo=totals['detection'] / images_seen,
                            loss_triplet=totals['triplet'] / images_seen, loss_critic=totals['critic'] / images_seen,
                            loss_generator=totals['adversarial'] / images_seen, peak_mem_GB=peak))
        audit = dict(epoch=epoch + 1, images_seen=images_seen, validation_images=val_count,
                     domain_images=domain_images, discriminator_updates=discriminator_updates,
                     valid_triplet_anchors=anchors, peak_mem_GB=peak,
                     experts_frozen=all(not p.requires_grad and p.grad is None for e in experts for p in e.parameters()))
        with (save / 'epoch_audit.jsonl').open('a') as f:
            f.write(json.dumps(audit) + '\n')
        print('EPOCH_COMPLETE ' + json.dumps(dict(**audit, val_loss=val_loss, best_val_loss=best)), flush=True)
    history.writer.close()
    atomic_json(save / 'completion.json', dict(epochs_completed=epochs, smoke_test=bool(args.smoke_steps), best_val_loss=best))
