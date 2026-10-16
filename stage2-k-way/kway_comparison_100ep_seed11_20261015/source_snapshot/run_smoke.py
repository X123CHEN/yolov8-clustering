"""Exercise three real updates for each planned 100-epoch experiment."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

SUITE = Path(__file__).resolve().parent.parent
BOOT = SUITE/'source_snapshot/run_experiment.py'
verification = json.loads((SUITE/'verification.json').read_text())
names = ['01_country_multi_unique', '02_latent_multi_unique', '03_country_kway_unique', '04_latent_kway_unique']


def run(gpu, name):
    alpha = verification['calibrations']['country' if 'country' in name else 'latent']['alpha_domain'] if 'kway' in name else .7
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    command = [sys.executable, '-u', str(BOOT), 'new_experiments.kway_domain.training',
               '--experiment', name, '--results_path', str(SUITE/'checks'), '--epochs', '100',
               '--seed', '11', '--batch_size', '12', '--num_workers', '8', '--smoke_steps', '3',
               '--alpha_domain', str(alpha)]
    with (SUITE/f'smoke_{gpu}.log').open('w') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(name, 'exit', process.returncode, flush=True)
    if process.returncode:
        raise RuntimeError(f'Smoke failed: {name}; see smoke_{gpu}.log')


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run, gpu, name) for gpu, name in enumerate(names)]
    for future in futures:
        future.result()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
with (SUITE/'check_smoke.log').open('w') as log:
    subprocess.run([sys.executable, '-u', str(BOOT), 'new_experiments.kway_domain.check_smoke', '--suite', str(SUITE)],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('ALL_SMOKE_CHECKS_PASSED', flush=True)
