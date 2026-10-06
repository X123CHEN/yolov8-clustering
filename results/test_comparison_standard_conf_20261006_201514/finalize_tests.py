import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'launch_manifest.json').read_text())
while True:
    completed = 0
    for run in manifest['runs']:
        status = Path(run['script']).with_suffix('.exit_code')
        if status.exists() and status.read_text().strip() != '0':
            message = {'status': 'failed', 'experiment': run['experiment'], 'log': run['log']}
            (root / 'evaluation_status.json').write_text(json.dumps(message, indent=2))
            raise SystemExit(str(message))
        completed += (root / run['experiment'] / 'completion.json').exists()
    if completed == len(manifest['runs']):
        break
    time.sleep(20)
result = subprocess.run([sys.executable, '-m', 'utils.summarize_comparison_tests', '--root', str(root)], cwd=manifest['project'])
original = root.parent / 'test_comparison_20261006'
expected = json.loads((root / 'original_results_sha256.json').read_text())
changed = [name for name, digest in expected.items()
           if not (original / name).is_file() or hashlib.sha256((original / name).read_bytes()).hexdigest() != digest]
(root / 'original_results_verification.json').write_text(json.dumps({'checked_files': len(expected), 'changed_files': changed, 'unchanged': not changed}, indent=2))
if changed:
    raise RuntimeError('Original results changed: ' + str(changed[:10]))
(root / 'evaluation_status.json').write_text(json.dumps({'status': 'complete' if result.returncode == 0 else 'summary_failed', 'exit_code': result.returncode}, indent=2))
raise SystemExit(result.returncode)
