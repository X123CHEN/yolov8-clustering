"""Controlled four-way ablation; shared baseline helpers are reused unchanged."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.yolo import YoloBody
from nets.yolo_training import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory, append_metrics_epoch
from utils.latent_validation import CocoValidationCallback
from utils.dataloader import YoloDataset_eval, yolo_dataset_collate_eval
from utils.utils import get_classes, seed_everything, worker_init_fn
from utils.comparison_training import (ROOT, DomainDataset, prepare_data, domain_triplet,
                                       file_hash, atomic_json, atomic_save, require_finite)
from utils.latent_membership import load_aligned
from utils.latent_adversarial import Adversary, adversarial_weight
from utils.latent_structure import source_conditions
from utils.latent_v2 import V2Adversary


class MembershipDataset(DomainDataset):
    def __init__(self, rows, records, owners, classes, epochs, memberships, conditions, return_indices=False):
        super().__init__(rows, records, owners, classes, epochs)
        self.memberships = memberships
        self.conditions = conditions
        self.return_indices = return_indices

    def __getitem__(self, index):
        return [(self.base[item], self.domains[item], self.memberships[item], self.conditions[item]) +
                ((item,) if self.return_indices else ()) for item in self.rows[index]]


def membership_collate(batch):
    samples = [item for row in batch for item in row]
    images, targets = yolo_dataset_collate_eval([item[0] for item in samples])
    values = (images, targets, torch.tensor([item[1] for item in samples], dtype=torch.long),
            torch.from_numpy(np.stack([item[2] for item in samples])),
            torch.tensor([item[3] for item in samples], dtype=torch.long))
    return values + ((torch.tensor([item[4] for item in samples]),) if len(samples[0]) == 5 else ())


def capture_sources(save, resume):
    files = [ROOT / 'train_latent_ablation.py', ROOT / 'latent_dg_config.py']
    for directory in ('nets', 'utils', 'core'):
        files.extend(sorted((ROOT / directory).glob('*.py')))
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = save / 'source_snapshot' / relative
        # Config is editable between runs; resume uses resolved_config from its run.
        if source.name == 'latent_dg_config.py' and resume:
            continue
        hashes[str(relative)] = file_hash(source)
        if not resume:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    hashes.pop('latent_dg_config.py', None)
    return hashes


def train(cfg, save, cache, smoke_steps=0, resume=False):
    v2 = cfg.get('adversarial_head') == 'pooled_mlp'
    source_ap_enabled = cfg.get('source_ap_enabled', True)
    # Frozen October 6 configurations predate this explicit field and used the
    # standard decoder. Never silently relabel those historical configurations.
    validation_protocol = cfg.get('validation', dict(metric='source_val_coco_ap50',
        confidence_mode='max_class_sigmoid', confidence=.05, nms_iou=.5, max_boxes=100, voc_fallback=False))
    args = SimpleNamespace(experiment=cfg['experiment'], epochs=cfg['epochs'], seed=cfg['seed'],
                           batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], smoke_steps=smoke_steps)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'run_config.json').exists() and not resume:
        raise FileExistsError(f'Use a fresh results_path; {save} already has a run')
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    names, train_path, val_path, rows, records, owners, val_lines, adv = prepare_data('latent_unique')
    group = len(rows[0])
    class_names, classes = get_classes(cfg["classes_path"])
    memberships, membership_prior = load_aligned(cache, records, owners, cfg['assignment'])
    conditions, conditional_prior, condition_info = source_conditions(
        records, owners, classes, cfg['num_domains'], cfg.get('conditional_prior_smoothing', 20.))
    confusion_target = cfg.get('kway_confusion_target', 'assignment_prior')
    prior = membership_prior
    if cfg['architecture'] == 'single_kway':
        if confusion_target == 'hard_prior':
            prior = np.bincount([owners[key] for key in sorted(records)], minlength=cfg['num_domains']).astype(np.float32)
            prior /= prior.sum()
        elif confusion_target == 'uniform':
            prior = np.full(cfg['num_domains'], 1 / cfg['num_domains'], dtype=np.float32)
    dataset = MembershipDataset(rows, records, owners, classes, args.epochs, memberships, conditions, return_indices=v2)
    online = None
    if v2 and cfg.get('center_mode') == 'online':
        from utils.online_latent_centers import OnlineCenters
        online = OnlineCenters(cfg, records, memberships)
    if args.smoke_steps:
        val_lines = val_lines[:args.batch_size + 1]
    validation = YoloDataset_eval(val_lines, [640, 640], classes, epoch_length=args.epochs,
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                  train=False, special_aug_ratio=0)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size // group, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=membership_collate,
                        worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed), generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=yolo_dataset_collate_eval)
    model = YoloBody([640, 640], classes, 's', pretrained=False).cuda()
    initial_path = Path(cfg["initial_weights"])
    initial = torch.load(initial_path, map_location='cpu')
    compatible = {k: v for k, v in initial.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    missing, _ = model.load_state_dict(compatible, strict=False)
    if any(k.startswith('backbone.') for k in missing):
        raise RuntimeError(f'Incomplete initial backbone: {missing}')
    # Capture before allocating domain-dependent modules: all four students must match.
    student_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    loss_fn = Loss(model)
    ema = ModelEMA(model)
    adversary = (V2Adversary if v2 else Adversary)(cfg, prior, conditional_prior)
    experts, discriminators, critic_opts = adversary.experts, adversary.discriminators, adversary.optimizers
    discriminator_initial_hashes = [hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
        for v in d.state_dict().values())).hexdigest() for d in discriminators]
    expert_info = adversary.expert_info
    pg0, pg1, pg2 = [], [], []
    for _, module in model.named_modules():
        if isinstance(getattr(module, 'bias', None), nn.Parameter):
            pg2.append(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            pg0.append(module.weight)
        elif isinstance(getattr(module, 'weight', None), nn.Parameter):
            pg1.append(module.weight)
    lr = min(max(args.batch_size / 64 * cfg["nominal_lr"], 5e-4), 0.05)
    min_lr = min(max(args.batch_size / 64 * cfg["nominal_lr"] * cfg["min_lr_ratio"], 5e-6), 0.0005)
    optimizer = torch.optim.SGD(pg0, lr, momentum=cfg["momentum"], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': cfg['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    scheduler_epochs = cfg.get('scheduler_epochs', args.epochs)
    schedule = get_lr_scheduler('cos', lr, min_lr, scheduler_epochs)
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
                  experts=expert_info, discriminators=len(discriminators), optimizer='SGD', momentum=cfg['momentum'],
                  weight_decay=cfg['weight_decay'], nominal_lr=cfg['nominal_lr'], fitted_lr=lr, min_lr=min_lr, scheduler='cos',
                  critic_optimizer='Adam', critic_lr=cfg['critic_lr'], critic_betas=cfg['critic_betas'],
                  input_shape=[640, 640], phi='s', fp16=False, freeze_train=False,
                  mosaic=False, mixup=False, fft=False, mixstyle=False, routing=cfg['assignment'],
                  detection_weight=cfg['detection_weight'], adversarial_weight=cfg['adv_weight'], triplet_weight=cfg['triplet_weight'],
                  adversarial_feature=cfg['adversarial_feature'], triplet_features=['feat3_20x20', 'feat2_40x40'],
                  triplet_inter_margin=0.5, triplet_intra_margin=0.1, triplet_intra_weight=0.1,
                  triplet_reduction='mean over valid pools; mean over scales; no duplicated K=2 global pool',
                  triplet_mining='class-based batch-hard in domain pools; exclude self/invalid anchors',
                  val_loss_reduction='image-weighted batch mean', best_criterion='minimum val_loss',
                  smoke_steps=args.smoke_steps, cluster1_expert_training='stopped during epoch 166; best epoch 52 authorized')
    source_hashes = capture_sources(save, resume)
    config.update(resolved_config=cfg, membership_cache=str(cache),
                  validation_protocol=validation_protocol,
                  membership_manifest_sha256=file_hash(cache / 'manifest.json'),
                  prior=prior.tolist(), membership_prior=membership_prior.tolist(), source_sha256=source_hashes,
                  discriminator_initial_sha256=discriminator_initial_hashes,
                  scheduler_epochs=scheduler_epochs, adv_warmup_epochs=cfg.get('adv_warmup_epochs', 0.0),
                  generator_bn_mode=cfg.get('generator_bn_mode', 'running'),
                  coco_period=cfg.get('coco_period', 1), loss_validation_period=1,
                  condition_source=condition_info,
                  condition_order_sha256=hashlib.sha256(conditions.tobytes()).hexdigest(),
                  expert_normalization=cfg.get('expert_normalization', 'batch'),
                  expert_reference_routing=cfg.get('expert_reference_routing', 'hard'),
                  expert_fake_assignment=cfg.get('expert_fake_assignment', 'follow'),
                  kway_confusion_target=confusion_target,
                  generator_reduction='sum_i sum_k q_ik * loss_ik / batch_size',
                  kway_generator='cross_entropy_to_fixed_training_prior',
                  update_order='G then D, detached pre-update G features; same as historical Hard-LD')
    if v2:
        config.update(adversarial_objective=cfg['adversarial_objective'], adversarial_head='P5 GAP MLP',
                      foreground_weight=cfg['foreground_weight'], source_ap_enabled=source_ap_enabled,
                      kway_generator=cfg['adversarial_objective'],
                      selection_metric='minimum source detection val_loss; seven-domain VOC50 for decisions')
    if cfg.get('kway_head') == 'conditional_style':
        config['kway_generator'] = 'cross_entropy_to_fixed_source_hard_conditional_prior'
        config['generator_reduction'] = 'mean source image conditional-prior cross entropy'
    if cfg.get('expert_reference_routing') == 'sample_soft':
        config['critic_reduction'] = 'sum sampled per-image real/fake loss / (2 B source_soft_prior[k])'
    if resume:
        previous = json.loads((save / 'run_config.json').read_text())
        for key in ('resolved_config', 'membership_manifest_sha256', 'train_order_sha256', 'validation_order_sha256',
                    'initial_sha256', 'student_initial_sha256', 'experts', 'source_sha256', 'train_sha256', 'val_sha256'):
            if config[key] != previous[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
    else:
        atomic_json(save / 'run_config.json', config)
        atomic_json(save / 'validation_protocol.json', dict(**validation_protocol,
            validation_images=len(val_lines), aggregation='pool source validation images; macro over classes',
            map_50_column='source_val_coco_ap50', coordinates='integer xyxy; score truncated to 6 characters',
            decoder_sha256=file_hash(ROOT / 'utils/utils_bbox.py'),
            validation_decoder_sha256=file_hash(ROOT / 'utils/latent_validation.py')))
    print('RUN_CONFIG ' + json.dumps(config, ensure_ascii=False), flush=True)
    if resume:
        from torch.utils.tensorboard import SummaryWriter
        history = LossHistory.__new__(LossHistory)
        history.log_dir = str(save / 'metrics')
        history.losses, history.val_loss = [], []
        history.writer = SummaryWriter(history.log_dir)
    else:
        history = LossHistory(str(save / 'metrics'), model, [640, 640])
    evaluator = CocoValidationCallback(model, [640, 640], class_names, classes, val_lines, str(save / 'metrics'), True,
                             confidence_mode=validation_protocol['confidence_mode'],
                             confidence=validation_protocol['confidence'], nms_iou=validation_protocol['nms_iou'],
                             max_boxes=validation_protocol['max_boxes'], letterbox_image=False,
                             map_out_path=str(save / 'temp_map_out'), eval_flag=source_ap_enabled and not args.smoke_steps and not resume, period=1)
    evaluator.period = cfg.get('coco_period', 1)
    best = float('inf')
    best_map = -float('inf')
    start_epoch = 0
    epochs = 1 if args.smoke_steps else args.epochs
    if resume:
        state = torch.load(save / 'training_state.pt', map_location='cpu')
        if state['config']['smoke_steps']:
            raise ValueError('Cannot resume a smoke test as a formal run')
        model.load_state_dict(state['model']); ema.ema.load_state_dict(state['ema'])
        ema.updates = state['ema_updates']; optimizer.load_state_dict(state['optimizer'])
        if v2:
            adversary.restore_control_state(state['adversarial_control'])
        if online:
            online.restore(state['online_centers']); adversary.set_prior(online.prior)
        for module, values in zip(discriminators, state['discriminators']): module.load_state_dict(values)
        for opt, values in zip(critic_opts, state['critic_optimizers']): opt.load_state_dict(values)
        adversary.restore_routing_state(state.get('routing_rng'))
        start_epoch, best = state['epoch'], state['best_val_loss']
        best_map = state.get('best_map_50', -float('inf'))
        if len((save / 'epoch_audit.jsonl').read_text().splitlines()) != start_epoch:
            raise ValueError('Epoch audit/state mismatch; inspect incomplete commit')
        coco_path = save / 'metrics/validation_coco_metrics.jsonl'
        evaluated = [json.loads(v) for v in coco_path.read_text().splitlines()] if coco_path.exists() else []
        scheduled = [v for v in evaluated if v.get('purpose', 'scheduled') == 'scheduled']
        if source_ap_enabled and [v['epoch'] for v in scheduled] != list(range(evaluator.period, start_epoch + 1, evaluator.period)):
            raise ValueError('Validation metrics/state mismatch; inspect incomplete commit')
        evaluator.maps = [0.] + [v['value'] for v in evaluated]
        evaluator.epoches = [0] + [v['epoch'] for v in evaluated]
        evaluator.last_map_50 = evaluator.maps[-1]
        evaluator.eval_flag = source_ap_enabled
        for name, attribute in [('epoch_loss.txt', 'losses'), ('epoch_val_loss.txt', 'val_loss')]:
            values = [float(v) for v in (save / 'metrics' / name).read_text().splitlines()]
            if len(values) != start_epoch:
                raise ValueError('Incomplete epoch commit; inspect logs before resuming')
            setattr(history, attribute, values)
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    for epoch in range(start_epoch, epochs):
        started = time.time()
        generator.manual_seed(args.seed + epoch)
        dataset.base.epoch_now = epoch
        model.train()
        set_optimizer_lr(optimizer, schedule, epoch)
        torch.cuda.reset_peak_memory_stats()
        totals = dict(total=0., detection=0., adversarial=0., triplet=0., critic=0., foreground=0.)
        images_seen, steps, anchors = 0, 0, 0
        domain_images = [0] * len(names)
        discriminator_updates = [0] * len(discriminators)
        membership_mass = torch.zeros(len(names), device='cuda')
        effective_steps = min(len(loader), args.smoke_steps) if args.smoke_steps else len(loader)
        first_gradient_norms, first_gradient_cosines, first_backbone_norms = {}, {}, {}
        sampled_counts = [0] * len(names)
        for batch in tqdm(itertools.islice(loader, effective_steps), total=effective_steps,
                                             desc=f'{args.experiment} epoch {epoch + 1}/{args.epochs}', mininterval=10):
            images, targets, domains, q, c = batch[:5]
            indices = batch[5].numpy() if v2 else None
            if online:
                q = torch.from_numpy(online.q[indices].copy())
            images, targets, domains = images.cuda(non_blocking=True), targets.cuda(non_blocking=True), domains.cuda()
            q = q.cuda(non_blocking=True).detach()
            c = c.cuda(non_blocking=True).detach()
            membership_mass += q.sum(0)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if v2:
                detection, foreground = loss_fn(outputs[:5], targets, return_foreground=True)
            else:
                detection = loss_fn(outputs[:5], targets)
                foreground = detection * 0
            adversarial = outputs[5].sum() * 0
            triplet, valid_anchors = adversarial, 0
            n = len(images)
            for domain in range(len(names)):
                domain_images[domain] += int((domains == domain).sum())
            adv_features = outputs[7] if cfg['adversarial_feature'] == 'feat1_style' else outputs[5]
            adversarial = (adversary.raw_generator_loss(adv_features, q) if v2
                           else adversary.generator_loss(adv_features, q, c))
            triplet, valid_anchors = domain_triplet(outputs, targets, domains, len(names))
            if v2:
                current_adv_weight = adversary.feedback(detection, adversarial, adv_features,
                    (epoch + steps / len(loader)) / epochs)
                task = detection + cfg['triplet_weight'] * triplet + cfg['foreground_weight'] * foreground
                total = task + adversary.student_term(adv_features, q, adversarial)
                sign = -1 if cfg['adversarial_objective'] == 'grl_domain_ce' else 1
                reported_total = task.detach() + sign * current_adv_weight * adversarial.detach()
            else:
                current_adv_weight = adversarial_weight(cfg, epoch, steps, len(loader))
                total = cfg['detection_weight'] * detection + current_adv_weight * adversarial + cfg['triplet_weight'] * triplet
                reported_total = total.detach()
            feature_gradient_norms = {}
            feature_gradient_cosines = {}
            if steps == 0 and not v2:
                feature_gradients = {}
                for name, value, weight in [('detection', detection, cfg['detection_weight']),
                                            ('adversarial', adversarial, current_adv_weight),
                                            ('triplet', triplet, cfg['triplet_weight'])]:
                    gradient = torch.autograd.grad(weight * value, adv_features, retain_graph=True, allow_unused=True)[0]
                    feature_gradient_norms[name] = float(gradient.norm()) if gradient is not None else 0.
                    feature_gradients[name] = gradient
                a, b = feature_gradients['detection'], feature_gradients['adversarial']
                if a is not None and b is not None:
                    feature_gradient_cosines['det_adv'] = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
                first_gradient_norms = dict(feature_gradient_norms)
                first_gradient_cosines = dict(feature_gradient_cosines)
                del feature_gradients, a, b
                if cfg.get('profile') == 'structural60' and (epoch % cfg.get('coco_period', 5) == 0 or args.smoke_steps):
                    parameters = tuple(model.backbone.parameters())
                    for name, value, weight in [('detection', detection, cfg['detection_weight']),
                                                ('adversarial', adversarial, current_adv_weight)]:
                        gradients = torch.autograd.grad(weight * value, parameters, retain_graph=True, allow_unused=True)
                        first_backbone_norms[name] = float(torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt())
                        del gradients
            require_finite(total, 'student loss')
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            critic, updates, critic_stats = adversary.update(images, adv_features, domains, q, c)
            if online and epoch >= cfg['center_freeze_epochs']:
                row = online.accumulate(indices, q.detach().cpu().numpy())
                if row is not None:
                    adversary.set_prior(online.prior)
                    with (save / 'online_centers.jsonl').open('a') as stream: stream.write(json.dumps(dict(epoch=epoch+1, **row))+'\n')
            sampled_counts = [a + b for a, b in zip(sampled_counts, critic_stats.get('sampled_reference_counts', [0] * len(names)))]
            discriminator_updates = [a + b for a, b in zip(discriminator_updates, updates)]
            for key, value in [('total', reported_total), ('detection', detection), ('adversarial', adversarial), ('triplet', triplet), ('foreground', foreground)]:
                totals[key] += float(value.detach()) * n
            totals['critic'] += critic * n
            anchors += valid_anchors
            images_seen += n
            steps += 1
            if steps == 1 or steps % 100 == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=steps, detection=float(detection.detach()),
                                      adversarial=float(adversarial.detach()), triplet=float(triplet.detach()),
                                      critic=critic, valid_triplet_anchors=valid_anchors, gradient_norm=float(norm),
                                      feature_gradient_norms=feature_gradient_norms,
                                      feature_gradient_cosines=feature_gradient_cosines,
                                      backbone_gradient_norms=first_backbone_norms if steps == 1 else {},
                                      effective_adv_weight=current_adv_weight, critic_stats=critic_stats,
                                      foreground=float(foreground.detach()), objective=cfg.get('adversarial_objective') if v2 else None)), flush=True)
        train_time = time.time() - started
        if online:
            row = online.commit()
            if row is not None:
                adversary.set_prior(online.prior)
                with (save / 'online_centers.jsonl').open('a') as stream: stream.write(json.dumps(dict(epoch=epoch+1, **row))+'\n')
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
        improved_val = val_loss < best
        if improved_val:
            best = val_loss
            atomic_save(ema.ema.state_dict(), save / 'best_epoch_weights.pth')
            atomic_json(save / 'best_checkpoint.json', dict(epoch=epoch + 1, val_loss=val_loss,
                                                          criterion='minimum val_loss', validation_images=val_count))
        atomic_save(ema.ema.state_dict(), save / 'last_epoch_weights.pth')
        history.append_loss(epoch + 1, totals['total'] / images_seen, val_loss)
        map50 = evaluator.on_epoch_end(epoch + 1, ema.ema)
        if improved_val and not args.smoke_steps:
            atomic_json(save / 'best_checkpoint.json', dict(epoch=epoch + 1, val_loss=val_loss,
                map_50=float(map50) if map50 is not None else None, metric='source_val_coco_ap50' if source_ap_enabled else None,
                weights_sha256=file_hash(save / 'best_epoch_weights.pth'),
                confidence_mode=validation_protocol['confidence_mode'],
                criterion='minimum val_loss', validation_images=val_count))
        improved_map = not args.smoke_steps and map50 is not None and float(map50) > best_map
        if improved_map:
            best_map = float(map50)
            atomic_save(ema.ema.state_dict(), save / 'best_map_epoch_weights.pth')
            atomic_json(save / 'best_map_checkpoint.json', dict(epoch=epoch + 1, map_50=best_map,
                metric='source_val_coco_ap50', confidence_mode=validation_protocol['confidence_mode'],
                criterion='maximum among scheduled source validation COCO AP50 measurements', validation_images=val_count))
        append_metrics_epoch(str(save / 'metrics'), dict(epoch=epoch + 1, train_loss=totals['total'] / images_seen,
                            val_loss=val_loss, map_50=map50, train_time_s=train_time, val_time_s=val_time,
                            epoch_time_s=time.time() - started, steps=steps, images_per_step=args.batch_size,
                            train_images_per_s=images_seen / train_time, loss_yolo=totals['detection'] / images_seen,
                            loss_triplet=totals['triplet'] / images_seen, loss_critic=totals['critic'] / images_seen,
                            loss_generator=totals['adversarial'] / images_seen, peak_mem_GB=peak))
        audit = dict(epoch=epoch + 1, images_seen=images_seen, validation_images=val_count,
                     domain_images=domain_images, discriminator_updates=discriminator_updates,
                     membership_mass=membership_mass.cpu().tolist(),
                     map_50=map50, best_map_50=best_map if np.isfinite(best_map) else None,
                     adversarial_weight_start=adversarial_weight(cfg, epoch, 0, len(loader)),
                     adversarial_weight_end=current_adv_weight,
                     feature_gradient_norms_first_batch=first_gradient_norms,
                     feature_gradient_cosines_first_batch=first_gradient_cosines,
                     backbone_gradient_norms_first_batch=first_backbone_norms,
                     feature_gradient_reference=cfg['adversarial_feature'], sampled_reference_counts=sampled_counts,
                     valid_triplet_anchors=anchors, peak_mem_GB=peak,
                     experts_frozen=all(not p.requires_grad and p.grad is None for e in experts for p in e.parameters()))
        if v2:
            audit.update(source_ap_enabled=source_ap_enabled, loss_foreground=totals['foreground']/images_seen,
                         feedback=adversary.last_feedback, objective=cfg['adversarial_objective'],
                         adversarial_weight_start=None, adversarial_weight_end=current_adv_weight)
        with (save / 'epoch_audit.jsonl').open('a') as f:
            f.write(json.dumps(audit) + '\n')
        # Keep optimizer/EMA/discriminator state so interrupted runs can be resumed faithfully.
        full_state = dict(epoch=epoch + 1, model=model.state_dict(), ema=ema.ema.state_dict(), ema_updates=ema.updates,
                         optimizer=optimizer.state_dict(), discriminators=[d.state_dict() for d in discriminators],
                         critic_optimizers=[o.state_dict() for o in critic_opts], best_val_loss=best, config=config,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         numpy_rng=np.random.get_state(), python_rng=random.getstate(), best_map_50=best_map,
                         routing_rng=adversary.routing_state())
        if v2:
            full_state['adversarial_control'] = adversary.control_state()
        if online:
            full_state['online_centers'] = online.state()
        atomic_save(full_state, save / 'training_state.pt')
        if improved_map:
            atomic_save(full_state, save / 'best_map_training_state.pt')
        if improved_val:
            atomic_save(full_state, save / 'best_val_training_state.pt')
        del full_state
        print('EPOCH_COMPLETE ' + json.dumps(dict(**audit, val_loss=val_loss, best_val_loss=best)), flush=True)
    # Evaluate the EXACT minimum-loss weights if that epoch was off the AP cadence.
    # Keep this replay separate from the scheduled curve and its auxiliary peak.
    if not args.smoke_steps and source_ap_enabled:
        selected = json.loads((save / 'best_checkpoint.json').read_text())
        if selected.get('map_50') is None:
            digest = file_hash(save / 'best_epoch_weights.pth')
            replay_path = save / 'selected_checkpoint_coco.json'
            if replay_path.exists():
                replay = json.loads(replay_path.read_text())
                if replay['epoch'] != selected['epoch'] or replay['weights_sha256'] != digest:
                    raise ValueError('Stale selected-checkpoint COCO replay')
                selected_map = replay['map_50']
            else:
                ema.ema.load_state_dict(torch.load(save / 'best_epoch_weights.pth', map_location='cpu'))
                ema.ema.eval()
                selected_map = evaluator.on_epoch_end(selected['epoch'], ema.ema, force=True,
                    purpose='selected_checkpoint_replay', weights_sha256=digest)
                replay = dict(epoch=selected['epoch'], map_50=selected_map, weights_sha256=digest,
                              metric='source_val_coco_ap50', confidence_mode=validation_protocol['confidence_mode'],
                              validation_images=len(validation), selection='minimum val_loss', evaluated_after_epoch=args.epochs)
                atomic_json(replay_path, replay)
            selected.update(map_50=float(selected_map), evaluation='selected_checkpoint_replay')
            atomic_json(save / 'best_checkpoint.json', selected)
    history.writer.close()
    atomic_json(save / 'completion.json', dict(epochs_completed=epochs, smoke_test=bool(args.smoke_steps),
        best_val_loss=best, best_map_50=best_map if np.isfinite(best_map) else None,
        source_ap_enabled=source_ap_enabled))
