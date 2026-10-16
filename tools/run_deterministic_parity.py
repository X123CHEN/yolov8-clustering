"""Test-only process wrapper: pin CUDA backend selection for migration parity.

Historical trainers explicitly enable cuDNN benchmarking after seeding. Pinning
the module properties in this isolated check process prevents timing-dependent
algorithm choices without editing the production recipe or its calibration.
"""
import os
from pathlib import Path
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
backend_type = type(torch.backends.cudnn)
backend_type.benchmark = property(lambda self: False, lambda self,value: None)
backend_type.deterministic = property(lambda self: True, lambda self,value: None)

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(script.parent))
print('PARITY_ONLY_BACKEND benchmark=False deterministic=True deterministic_algorithms=True',flush=True)
runpy.run_path(str(script),run_name='__main__')
