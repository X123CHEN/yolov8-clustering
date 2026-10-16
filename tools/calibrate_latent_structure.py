"""One source-only D calibration; never optimize or evaluate the detector.

The coefficient rule is fixed before seeing any AP: median weighted backbone
gradient ratio 3%, capped at 10% for every observed batch. Both K-way methods
share one coefficient, chosen against the larger measured paired-head ratio.
This is scale calibration at the initial detector, not an AP hyperparameter search.
"""
import argparse
from datetime import datetime
from functools import partial
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--critic-steps', type=int, default=64)
    parser.add_argument('--reference-weights', type=Path,
                        help='Source-trained detector stress point only; NEVER a formal initialization')
    parser.add_argument('--initial-calibration', type=Path,
                        help='Combine both observed regimes under the same gradient-scale rule')
    args = parser.parse_args()
    if args.batches < 8 or args.critic_steps < args.batches:
        parser.error('Use at least eight fixed batches and one complete D pass')
    if bool(args.reference_weights) != bool(args.initial_calibration):
        parser.error('Reference stress point and initial calibration must be supplied together')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import latent_dg_config as settings
    from nets.yolo import YoloBody
    from nets.yolo_training import Loss
    from utils.comparison_training import prepare_data, file_hash, atomic_json
    from utils.latent_ablation_training import MembershipDataset, membership_collate, capture_sources
    from utils.latent_adversarial import Adversary
    from utils.latent_structure import source_conditions
    from utils.latent_membership import ensure_cache, load_aligned
    from utils.utils import get_classes, seed_everything, worker_init_fn

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one idle GPU')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source_hashes = capture_sources(out,False)
    source_hashes['tools/calibrate_latent_structure.py'] = file_hash(__file__)
    configs = {n:settings.resolve(n,'structural60') for n in ['soft_expert','hard_kway','soft_kway']}
    _,train_path,val_path,rows,records,owners,_,_ = prepare_data('latent_unique')
    _,classes = get_classes(configs['soft_expert']['classes_path'])
    evidence = dict(created_at=datetime.now().astimezone().isoformat(), profile='structural60',
        source_sha256=source_hashes, train_sha256=file_hash(train_path), val_sha256=file_hash(val_path),
        detector_optimizer_steps=0, detector_ap_evaluations=0, target_evaluation=False,
        batches=args.batches, critic_steps=args.critic_steps, measured={},
        rule=dict(median_target_ratio=.03,observed_batch_max_ratio=.10,paired_kway_same_coefficient=True),
        limitations=['Initial detector with D-only warmup on repeated source batches; not the trained trajectory.',
                     'Gradient calibration constrains observed scale and does not predict AP or generalization.',
                     'All calibration model, BN, D and routing states are discarded before formal training.'])
    initial_evidence = None
    if args.initial_calibration:
        initial_evidence = json.loads(args.initial_calibration.read_text())
        if not initial_evidence['complete'] or initial_evidence['batches'] != args.batches or initial_evidence['critic_steps'] != args.critic_steps:
            raise ValueError('Incomparable initial calibration')
        # The diagnostic runner itself changed to add a second regime; auxiliary
        # heads, data, detector code and coefficient rule must remain identical.
        for path,digest in initial_evidence['source_sha256'].items():
            if path != 'tools/calibrate_latent_structure.py' and file_hash(ROOT/path) != digest:
                raise ValueError(f'Core code changed between calibration regimes: {path}')
        evidence.update(initial_calibration=str(args.initial_calibration.resolve()),
                        initial_calibration_sha256=file_hash(args.initial_calibration),
                        reference_weights=str(args.reference_weights.resolve()),
                        reference_weights_sha256=file_hash(args.reference_weights),
                        reference_is_formal_initialization=False,
                        reason_for_second_regime='Initial-only K-way coefficients were about 15; initial gradient scale is insufficient to constrain trained-detector behavior.')
    atomic_json(out/'calibration_protocol.json',evidence)
    for name,cfg in configs.items():
        seed_everything(cfg['seed'])
        torch.backends.cudnn.benchmark = True
        cache,_ = ensure_cache(cfg['membership'])
        q,prior = load_aligned(cache,records,owners,cfg['assignment'])
        c,table,info = source_conditions(records,owners,classes,cfg['num_domains'],cfg['conditional_prior_smoothing'])
        ds = MembershipDataset(rows,records,owners,classes,cfg['epochs'],q,c)
        loader = DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,drop_last=False,
            num_workers=cfg['num_workers'],collate_fn=membership_collate,
            worker_init_fn=partial(worker_init_fn,rank=0,seed=cfg['seed']),
            generator=torch.Generator().manual_seed(cfg['seed']))
        model = YoloBody([640,640],classes,'s',pretrained=False).cuda()
        weights = torch.load(cfg['initial_weights'],map_location='cpu')
        model.load_state_dict({k:v for k,v in weights.items() if k in model.state_dict() and v.shape==model.state_dict()[k].shape},strict=False)
        initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        baseline = json.loads((ROOT/'results/comparison_20261005_100ep/latent_unique/run_config.json').read_text())
        if initial != baseline['student_initial_sha256']:
            raise ValueError('Calibration detector differs from formal historical initialization')
        if args.reference_weights:
            model.load_state_dict(torch.load(args.reference_weights,map_location='cpu'),strict=True)
        loss_fn = Loss(model)
        if cfg['architecture']=='single_kway':prior=np.bincount([owners[k] for k in sorted(records)],minlength=2).astype(np.float32)/len(records)
        adversary = Adversary(cfg,prior,table)
        frozen = []
        model.train()
        index = 7 if cfg['adversarial_feature']=='feat1_style' else 5
        for images,targets,domains,qs,conditions in itertools.islice(loader,args.batches):
            with torch.no_grad():
                features=model(images.cuda())[index].cpu()
            frozen.append((images,targets,domains,qs,conditions,features))
        if len(frozen)!=args.batches:raise ValueError('Insufficient calibration batches')
        image_hashes = [hashlib.sha256(x[0].numpy().tobytes()).hexdigest() for x in frozen]
        initial_d = [hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in d.state_dict().values())).hexdigest() for d in adversary.discriminators]
        critic_trace=[]
        for step in range(args.critic_steps):
            images,_,domains,qs,conditions,features=frozen[step%len(frozen)]
            value,_,stats=adversary.update(images.cuda(),features.cuda(),domains.cuda(),qs.cuda(),conditions.cuda())
            if step in (0,args.critic_steps-1):critic_trace.append(dict(step=step+1,loss=value,stats=stats))
        measured=[]
        parameters=tuple(model.backbone.parameters())
        for number,(images,targets,_,qs,conditions,_) in enumerate(frozen,1):
            outputs=model(images.cuda())
            det=loss_fn(outputs[:5],targets.cuda())
            adv=adversary.generator_loss(outputs[index],qs.cuda(),conditions.cuda())
            det_g=torch.autograd.grad(det,parameters,retain_graph=True,allow_unused=True)
            adv_g=torch.autograd.grad(adv,parameters,allow_unused=True)
            det_n=torch.stack([g.detach().square().sum() for g in det_g if g is not None]).sum().sqrt()
            adv_n=torch.stack([g.detach().square().sum() for g in adv_g if g is not None]).sum().sqrt()
            dot=sum((a*b).sum() for a,b in zip(det_g,adv_g) if a is not None and b is not None)
            ratio=float(adv_n/det_n)
            if not np.isfinite(ratio) or ratio<=0:raise ValueError('Invalid calibration gradient ratio')
            measured.append(dict(batch=number,det_loss=float(det.detach()),adv_loss=float(adv.detach()),
                detection_backbone_norm=float(det_n),raw_adv_backbone_norm=float(adv_n),
                raw_ratio=ratio,gradient_cosine=float(dot/(det_n*adv_n))))
            del outputs,det,adv,det_g,adv_g,dot,det_n,adv_n
        ratios=[v['raw_ratio'] for v in measured]
        evidence['measured'][name]=dict(config=cfg,student_initial_sha256=initial,
            initial_discriminator_sha256=initial_d,source_image_tensor_sha256=image_hashes,
            membership_manifest_sha256=file_hash(cache/'manifest.json'),condition_source=info,
            prior=prior.tolist(),gradients=measured,raw_ratio_median=float(np.median(ratios)),
            raw_ratio_max=max(ratios),critic_trace=critic_trace)
        atomic_json(out/'calibration_progress.json',evidence)
        print('CALIBRATED',name,evidence['measured'][name]['raw_ratio_median'],flush=True)
        del model,loss_fn,adversary,parameters,frozen,weights,loader,ds,features
        gc.collect();torch.cuda.empty_cache()
    a,b=[evidence['measured'][n] for n in ['hard_kway','soft_kway']]
    if a['source_image_tensor_sha256']!=b['source_image_tensor_sha256'] or a['initial_discriminator_sha256']!=b['initial_discriminator_sha256'] or a['prior']!=b['prior']:
        raise ValueError('Paired K-way calibration differs beyond D targets')
    def coefficient(entries):
        return min(.03/max(x['raw_ratio_median'] for x in entries),.10/max(x['raw_ratio_max'] for x in entries))
    expert_entries=[evidence['measured']['soft_expert']]
    paired_entries=[a,b]
    if initial_evidence:
        for name in configs:
            if evidence['measured'][name]['source_image_tensor_sha256'] != initial_evidence['measured'][name]['source_image_tensor_sha256']:
                raise ValueError('Source calibration samples changed between detector regimes')
        expert_entries.append(initial_evidence['measured']['soft_expert'])
        paired_entries.extend(initial_evidence['measured'][n] for n in ['hard_kway','soft_kway'])
    evidence['recommended_adv_weight']=dict(soft_expert=coefficient(expert_entries),
        hard_kway=coefficient(paired_entries),soft_kway=coefficient(paired_entries))
    if any(file_hash(ROOT/p)!=h for p,h in source_hashes.items()):raise ValueError('Source changed during calibration')
    evidence['complete']=True
    atomic_json(out/'calibration.json',evidence)
    print(json.dumps(evidence['recommended_adv_weight']),flush=True)


if __name__=='__main__':main()
