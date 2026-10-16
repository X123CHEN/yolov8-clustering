"""Launch one detached screen per idle GPU for the four evaluation jobs."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'launch_manifest.json').read_text())
query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
screens = subprocess.run(['screen', '-ls'], text=True, capture_output=True).stdout
states = {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]] for row in query.splitlines()}
for run in manifest['runs']:
    if run['screen'] in screens:
        continue
    memory, utilization = states[run['gpu']]
    if memory > 100 or utilization > 0:
        raise RuntimeError(f"GPU {run['gpu']} is occupied: {memory} MiB, {utilization}%")
for run in manifest['runs']:
    if run['screen'] in screens:
        print('RUNNING', run['experiment'], 'GPU', run['gpu'], run['screen'], flush=True)
        continue
    subprocess.run(['screen', '-dmS', run['screen'], 'bash', run['script']], check=True)
    print('STARTED', run['experiment'], 'GPU', run['gpu'], run['screen'], flush=True)
print(subprocess.check_output(['screen', '-ls'], text=True))
