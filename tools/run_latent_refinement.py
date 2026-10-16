"""Three independent screen jobs, source-validation only; no test/evaluation stage."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import latent_dg_config as settings
from utils.summarize_latent_training import atomic_json,summarize

NAMES = ['soft_expert','hard_kway','soft_kway']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_evidence():
    folder = ROOT/'results/comparison_20261005_100ep/latent_unique'
    config = json.loads((folder/'run_config.json').read_text())
    metrics = list(csv.DictReader((folder/'metrics/metrics_epoch.csv').open()))
    if [int(r['epoch']) for r in metrics]!=list(range(1,101)):
        raise ValueError('Historical baseline must have all 100 epochs')
    peak = max(metrics,key=lambda r:float(r['map_50']))
    early = max((r for r in metrics if int(r['epoch'])<=60),key=lambda r:float(r['map_50']))
    checkpoint = json.loads((folder/'best_checkpoint.json').read_text())
    selected = min(metrics,key=lambda r:float(r['val_loss']))
    if checkpoint['epoch']!=int(selected['epoch']) or abs(checkpoint['val_loss']-float(selected['val_loss']))>1e-12:
        raise ValueError('Historical best checkpoint is inconsistent with minimum validation loss')
    for path_key,hash_key in [('initial_weights','initial_sha256'),('train_txt','train_sha256'),('val_txt','val_sha256')]:
        if sha(config[path_key])!=config[hash_key]: raise ValueError(f'Changed baseline input: {path_key}')
    for name in NAMES:
        c = settings.resolve(name)
        if c['epochs']!=60 or c['scheduler_epochs']!=100: raise ValueError('Expected 60 epochs, original 100-epoch LR horizon')
        for key,old_key in [('seed','seed'),('batch_size','total_batch_size'),('num_workers','num_workers'),
            ('nominal_lr','nominal_lr'),('momentum','momentum'),('weight_decay','weight_decay'),
            ('initial_weights','initial_weights'),('detection_weight','detection_weight'),('triplet_weight','triplet_weight')]:
            if c[key]!=config[old_key]: raise ValueError(f'{name}: non-comparable {key}')
        if c['validation']!=settings.VALIDATION or c['validation']['confidence_mode']!='legacy_uncertainty':
            raise ValueError('Historical metric protocol is required')
    return dict(training_path=str(folder),metrics_sha256=sha(folder/'metrics/metrics_epoch.csv'),
        config_sha256=sha(folder/'run_config.json'),peak_map_50=float(peak['map_50']),peak_epoch=int(peak['epoch']),
        selection='minimum source validation detection loss',selected_map_50=float(selected['map_50']),
        selected_epoch=int(selected['epoch']),selected_val_loss=float(selected['val_loss']),
        selected_weights_sha256=sha(folder/'best_epoch_weights.pth'),
        first60_peak_map_50=float(early['map_50']),first60_peak_epoch=int(early['epoch']),
        validation_protocol=dict(settings.VALIDATION),student_initial_sha256=config['student_initial_sha256'])


def verify_sources(manifest):
    verification=json.loads(Path(manifest['verification']).read_text())
    if not verification['passed']: raise ValueError('Preflight failed')
    for name,digest in verification['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Code changed after preflight: {name}')


def worker(suite,name):
    manifest=json.loads((suite/'manifest.json').read_text())
    lock=(suite/f'{name}.worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources(manifest)
    config_path=suite/f'{name}.json'
    if sha(config_path)!=manifest['config_sha256'][name]: raise ValueError('Frozen configuration changed')
    cfg=settings.validate(json.loads(config_path.read_text()))
    folder=suite/'training'/name
    gpu=manifest['gpus'][name]

    def update(phase,**extra):
        with (suite/'status.lock').open('a') as state_lock:
            fcntl.flock(state_lock,fcntl.LOCK_EX)
            state=json.loads((suite/'status.json').read_text())
            state['jobs'][name].update(phase=phase,pid=os.getpid(),screen=os.environ.get('STY'),**extra)
            phases=[v['phase'] for v in state['jobs'].values()]
            state['status']='complete' if all(p=='complete' for p in phases) else ('running_with_failures' if 'failed' in phases else 'running')
            atomic_json(suite/'status.json',state)

    process=None
    try:
        if not (folder/'completion.json').exists():
            if (folder/'run_config.json').exists():
                if not (folder/'training_state.pt').exists(): raise ValueError('Uncommitted first epoch; inspect before restarting')
                command=[sys.executable,'-u',str(ROOT/'train_latent_ablation.py'),'--resume',str(folder)]
            else:
                command=[sys.executable,'-u',str(ROOT/'train_latent_ablation.py'),'--resolved-config',str(config_path),'--results-path',str(folder)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (suite/f'{name}_train.log').open('ab',buffering=0) as log:
                log.write(('\nTRAIN_ONLY_LAUNCH '+json.dumps(dict(command=command,gpu=gpu,screen=os.environ.get('STY')))+'\n').encode())
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                update('training',train_pid=process.pid)
                while process.poll() is None:
                    summarize(suite)
                    time.sleep(30)
                if process.returncode: raise subprocess.CalledProcessError(process.returncode,command)
        completion=json.loads((folder/'completion.json').read_text())
        if completion['smoke_test'] or completion['epochs_completed']!=cfg['epochs']: raise ValueError('Incomplete formal training')
        actual=json.loads((folder/'run_config.json').read_text())
        if actual['student_initial_sha256']!=manifest['baseline']['student_initial_sha256']: raise ValueError('Student initialization changed')
        update('complete',train_pid=None)
        result=summarize(suite)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        # Do not leave an orphan training job if the supervisor itself fails.
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        update('failed',error=repr(exc),train_pid=None)
        raise


def launch(output,verification,gpus):
    if len(set(gpus))!=3: raise ValueError('Three distinct GPUs required')
    proof=json.loads(verification.read_text())
    if not proof['passed']: raise ValueError('Preflight must pass')
    baseline=baseline_evidence()
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(created_at=stamp,experiments=NAMES,baseline=baseline,verification=str(verification.resolve()),
        gpus=dict(zip(NAMES,gpus)),config_sha256={},test_evaluation=False)
    verify_sources(manifest)
    for n in NAMES:
        c=settings.resolve(n)
        proven=json.loads((Path(proof['smoke_runs'][n])/'run_config.json').read_text())['resolved_config']
        if c != proven:
            raise ValueError(f'{n}: configuration changed since successful preflight')
        c['gpu']=manifest['gpus'][n]
        atomic_json(output/f'{n}.json',c);manifest['config_sha256'][n]=sha(output/f'{n}.json')
    manifest['gpu_override_only']=True
    (output/'latent_dg_config.py').write_bytes((ROOT/'latent_dg_config.py').read_bytes())
    atomic_json(output/'manifest.json',manifest)
    if proof.get('acceptance_criteria'):
        atomic_json(output/'acceptance_criteria.json',proof['acceptance_criteria'])
    state=dict(status='starting',jobs={n:dict(phase='pending',gpu=manifest['gpus'][n]) for n in NAMES})
    atomic_json(output/'status.json',state)
    summarize(output)
    for n in NAMES:
        session=f"latent60_{n}_g{manifest['gpus'][n]}_{stamp}"
        subprocess.run(['screen','-L','-Logfile',str(output/f'{n}_worker.log'),'-dmS',session,
            sys.executable,'-u',str(Path(__file__).resolve()),'--suite',str(output),'--worker',n],check=True)
        print(session,flush=True)
    print('SUITE',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verification',type=Path)
    parser.add_argument('--gpus',nargs=3,default=['5','1','2'])
    parser.add_argument('--suite',type=Path)
    parser.add_argument('--worker',choices=NAMES)
    args=parser.parse_args()
    if args.worker:
        if not args.suite: parser.error('--worker requires --suite')
        worker(args.suite.resolve(),args.worker)
    else:
        if not args.output or not args.verification: parser.error('launch requires --output and --verification')
        launch(args.output.resolve(),args.verification.resolve(),args.gpus)


if __name__=='__main__': main()
