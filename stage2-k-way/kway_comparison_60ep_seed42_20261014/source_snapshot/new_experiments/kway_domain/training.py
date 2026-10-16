"""Four paired Unique experiments with the stage-one detection and FAB semantics."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from nets.yolo import YoloBody
from nets.backbone import Backbone_pre
from nets.maddg import Discriminator1
from nets.yolo_training import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory, EvalCallback, append_metrics_epoch
from utils.dataloader import YoloDataset_eval, yolo_dataset_collate_eval
from utils.utils import get_classes, seed_everything, worker_init_fn, init_weights
from utils.comparison_training import (ROOT, EXPERT_ROOT, prepare_data, DomainDataset,
    domain_collate, domain_triplet, require_finite, file_hash, atomic_json, atomic_save)
from new_experiments.kway_domain.discriminator import (KWayDiscriminator,
    hard_domain_loss, dann_coefficient, confusion_counts)
from new_experiments.kway_domain.cluster_index import audit_source_labels

EXPERIMENTS = {
    '01_country_multi_unique': ('country_unique', False),
    '02_latent_multi_unique': ('latent_unique', False),
    '03_country_kway_unique': ('country_unique', True),
    '04_latent_kway_unique': ('latent_unique', True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--results_path', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=12, help='Total student images per batch, not per domain')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--smoke_steps', type=int, default=0)
    parser.add_argument('--alpha_domain', type=float, default=0.7)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size < 2 or args.batch_size % 6 or args.num_workers < 0 or args.smoke_steps < 0 or args.alpha_domain < 0:
        parser.error('Use positive epochs, a total batch size divisible by 6, and nonnegative workers/smoke_steps')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    save = args.results_path.resolve() / args.experiment
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'run_config.json').exists() and not args.resume:
        raise FileExistsError(f'Use a fresh results_path; {save} already has a run')
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    data_experiment, kway = EXPERIMENTS[args.experiment]
    label_audit = audit_source_labels()
    names, train_path, val_path, rows, records, owners, val_lines, adv = prepare_data(data_experiment)
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
    loss_fn = Loss(model)
    ema = ModelEMA(model)
    experts, discriminators, critic_opts, expert_info = [], [], [], []
    if not kway:
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
    else:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(args.seed + 100)
            discriminator = KWayDiscriminator(len(names), input_channels=512).cuda()
            init_weights(discriminator, 'xavier')
        discriminators.append(discriminator)
        critic_opts.append(torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999)))
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
                  initial_weights=str(initial_path), initial_sha256=file_hash(initial_path),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), domains=names,
                  experts=expert_info, discriminators=len(discriminators), optimizer='SGD', momentum=0.937,
                  weight_decay=5e-4, nominal_lr=0.01, fitted_lr=lr, min_lr=min_lr, scheduler='cos',
                  critic_optimizer='Adam', critic_lr=1e-5, critic_betas=[0.9, 0.999],
                  input_shape=[640, 640], phi='s', fp16=False, freeze_train=False,
                  mosaic=False, mixup=False, fft=False, mixstyle=False, routing='hard_domain_id',
                  detection_weight=1.0, adversarial_weight=args.alpha_domain if kway else 0.7, triplet_weight=0.3,
                  adversarial_feature='feat3_20x20', triplet_features=['feat3_20x20', 'feat2_40x40'],
                  triplet_inter_margin=0.5, triplet_intra_margin=0.1, triplet_intra_weight=0.1,
                  triplet_reduction='mean over valid pools; mean over scales; no duplicated K=2 global pool',
                  triplet_mining='class-based batch-hard in domain pools; exclude self/invalid anchors',
                  val_loss_reduction='image-weighted batch mean', best_criterion='minimum val_loss',
                  smoke_steps=args.smoke_steps, architecture='single_kway' if kway else 'expert_multi',
                  alpha_domain=args.alpha_domain if kway else 0.7,
                  domain_loss='hard_ce_mean' if kway else 'historical_lsgan',
                  grl_schedule='dann_gamma10_optimizer_step' if kway else None,
                  update_order='joint_backward_then_student_and_D_step' if kway else 'student_then_D',
                  scheduler_epochs=args.epochs, data_protocol='unique', label_audit=label_audit,
                  domain_counts=[sum(value == k for value in owners.values()) for k in range(len(names))],
                  majority_accuracy_reference=max(sum(value == k for value in owners.values()) for k in range(len(names))) / len(records),
                  balanced_accuracy_reference=1 / len(names),
                  prior_entropy=float(-sum((sum(value == k for value in owners.values()) / len(records)) * np.log(sum(value == k for value in owners.values()) / len(records)) for k in range(len(names)))),
                  head_parameters=sum(p.numel() for d in discriminators for p in d.parameters()),
                  expert_parameters=sum(p.numel() for e in experts for p in e.parameters()),
                  student_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters()) + sum(p.numel() for d in discriminators for p in d.parameters()),
                  total_training_parameters=sum(p.numel() for p in model.parameters()) + sum(p.numel() for d in discriminators for p in d.parameters()) + sum(p.numel() for e in experts for p in e.parameters()))
    starting_epoch, global_step = 0, 0
    best = float('inf')
    if args.resume:
        state = torch.load(save / 'training_state.pt', map_location='cuda')
        previous = state['config']
        for key in ('experiment', 'epochs', 'seed', 'alpha_domain', 'initial_sha256', 'train_sha256', 'val_sha256'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(state['model'])
        ema.ema.load_state_dict(state['ema']); ema.updates = state['ema_updates']
        optimizer.load_state_dict(state['optimizer'])
        for d, saved in zip(discriminators, state['discriminators']): d.load_state_dict(saved)
        for opt, saved in zip(critic_opts, state['critic_optimizers']): opt.load_state_dict(saved)
        starting_epoch, global_step, best = state['epoch'], state['global_step'], state['best_val_loss']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    atomic_json(save / 'run_config.json', config)
    print('RUN_CONFIG ' + json.dumps(config, ensure_ascii=False), flush=True)
    if args.resume:
        # LossHistory creates its directory; reopen the existing writer on resume.
        from torch.utils.tensorboard import SummaryWriter
        history = LossHistory.__new__(LossHistory)
        history.log_dir = str(save / 'metrics')
        history.losses = [float(x) for x in (save / 'metrics/epoch_loss.txt').read_text().split()]
        history.val_loss = [float(x) for x in (save / 'metrics/epoch_val_loss.txt').read_text().split()]
        history.writer = SummaryWriter(history.log_dir)
    else:
        history = LossHistory(str(save / 'metrics'), model, [640, 640])
    evaluator = EvalCallback(model, [640, 640], class_names, classes, val_lines, str(save / 'metrics'), True,
                             map_out_path=str(save / 'temp_map_out'), eval_flag=not args.smoke_steps, period=1)
    epochs = 1 if args.smoke_steps else args.epochs
    for epoch in range(starting_epoch, epochs):
        started = time.time()
        generator.manual_seed(args.seed + epoch)
        dataset.base.epoch_now = epoch
        model.train()
        set_optimizer_lr(optimizer, schedule, epoch)
        torch.cuda.reset_peak_memory_stats()
        totals = dict(total=0., detection=0., adversarial=0., triplet=0., critic=0.)
        images_seen, steps, anchors = 0, 0, 0
        domain_images = [0] * len(names)
        discriminator_updates = [0] * len(discriminators)
        confusion = torch.zeros(len(names), len(names), dtype=torch.long, device='cuda')
        probability_sum = torch.zeros(len(names), device='cuda')
        domain_confidence_sum = 0.
        domain_entropy_sum = 0.
        gradient_checks = []
        effective_steps = min(len(loader), args.smoke_steps) if args.smoke_steps else len(loader)
        for images, targets, domains in tqdm(itertools.islice(loader, effective_steps), total=effective_steps,
                                             desc=f'{args.experiment} epoch {epoch + 1}/{args.epochs}', mininterval=10):
            images, targets, domains = images.cuda(non_blocking=True), targets.cuda(non_blocking=True), domains.cuda()
            optimizer.zero_grad(set_to_none=True)
            if kway:
                discriminators[0].train().requires_grad_(True)
                critic_opts[0].zero_grad(set_to_none=True)
            outputs = model(images)
            detection = loss_fn(outputs[:5], targets)
            adversarial = outputs[5].sum() * 0
            triplet, valid_anchors = adversarial, 0
            n = len(images)
            for domain in range(len(names)):
                domain_images[domain] += int((domains == domain).sum())
            grl_coeff = dann_coefficient(global_step, args.epochs * len(loader)) if kway else 0.
            if kway:
                adversarial, logits = hard_domain_loss(discriminators[0], outputs[5], domains, grl_coeff)
                if logits.shape != (n, len(names)):
                    raise RuntimeError(f'Invalid K-way logits shape: {logits.shape}')
                with torch.no_grad():
                    confusion += confusion_counts(logits, domains)
                    probabilities = logits.softmax(1)
                    probability_sum += probabilities.sum(0)
                    domain_confidence_sum += float(probabilities.max(1).values.sum())
                    domain_entropy_sum += float(-(probabilities * probabilities.clamp_min(1e-8).log()).sum())
            else:
                for domain, discriminator in enumerate(discriminators):
                    discriminator.eval().requires_grad_(False)
                    mask = domains == domain
                    if mask.any():
                        prediction = discriminator(outputs[5][mask])
                        adversarial = adversarial + (prediction - 1).square().flatten(1).mean(1).sum() / n
            triplet, valid_anchors = domain_triplet(outputs, targets, domains, len(names))
            domain_weight = args.alpha_domain if kway else 0.7
            total = detection + domain_weight * adversarial + 0.3 * triplet
            if steps < (3 if args.smoke_steps else 1):
                norms = {}
                for key, value in [('detection', detection), ('fab', 0.3 * triplet), ('domain', domain_weight * adversarial)]:
                    gradient = torch.autograd.grad(value, outputs[5], retain_graph=True, allow_unused=True)[0]
                    norms[key] = float(gradient.norm()) if gradient is not None else 0.
                norms.update(epoch=epoch + 1, step=global_step, grl_coeff=grl_coeff,
                             domain_det_ratio=norms['domain'] / max(norms['detection'], 1e-12))
                gradient_checks.append(norms)
                with (save / 'gradient_checks.jsonl').open('a') as stream: stream.write(json.dumps(norms) + '\n')
            require_finite(total, 'student loss')
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            if kway:
                critic_opts[0].step()
                discriminator_updates[0] += 1
            ema.update(model)
            critic = float(adversarial.detach()) if kway else 0.
            if not kway:
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
            global_step += 1
            if steps == 1 or steps % 100 == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=steps, detection=float(detection.detach()),
                                      adversarial=float(adversarial.detach()), triplet=float(triplet.detach()),
                                      critic=critic, valid_triplet_anchors=valid_anchors, gradient_norm=float(norm),
                                      grl_coeff=grl_coeff, alpha_domain=domain_weight,
                                      domain_valid_samples=n, domain_supervision_ratio=1.)), flush=True)
        torch.cuda.synchronize()
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
                         numpy_rng=np.random.get_state(), python_rng=random.getstate(), global_step=global_step,
                         scheduler=dict(kind='cos', epochs=args.epochs, lr=lr, min_lr=min_lr)), save / 'training_state.pt')
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
                     experts_frozen=all(not p.requires_grad and p.grad is None for e in experts for p in e.parameters()),
                     domain_supervision_ratio=1., grl_coeff=grl_coeff, alpha_domain=domain_weight,
                     gradient_checks=gradient_checks)
        samples = []
        for index in random.Random(args.seed + epoch).sample(range(len(dataset.domains)), 3):
            path = sorted(records)[index]
            country = Path(path).parent.parent.name
            samples.append(dict(image_id=Path(path).stem, path=path, country=country, domain_id=owners[path]))
        audit['label_samples'] = samples
        if kway:
            recalls = confusion.diag().float() / confusion.sum(1).clamp_min(1)
            audit.update(domain_confusion=confusion.cpu().tolist(), per_domain_recall=recalls.cpu().tolist(),
                         domain_balanced_accuracy=float(recalls.mean()),
                         domain_accuracy=float(confusion.diag().sum()) / images_seen,
                         predicted_distribution=(confusion.sum(0).float() / images_seen).cpu().tolist(),
                         probability_mean=(probability_sum / images_seen).cpu().tolist(),
                         mean_confidence=domain_confidence_sum / images_seen,
                         mean_prediction_entropy=domain_entropy_sum / images_seen)
        with (save / 'epoch_audit.jsonl').open('a') as f:
            f.write(json.dumps(audit) + '\n')
        print('EPOCH_COMPLETE ' + json.dumps(dict(**audit, val_loss=val_loss, best_val_loss=best)), flush=True)
    history.writer.close()
    atomic_json(save / 'completion.json', dict(epochs_completed=epochs, smoke_test=bool(args.smoke_steps), best_val_loss=best))


if __name__ == '__main__':
    main()
