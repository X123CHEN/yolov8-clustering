"""Frozen protocol for the source-only latent domain analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEIGHTS = PROJECT / 'model_data/yolov8_s.pth'
MANIFEST_ROOT = PROJECT / 'datasets/RDD2022/ImageSets'
COUNTRIES = ('India', 'Japan', 'United_States')
EXPECTED_COUNTS = (2578, 6320, 3844)
FEATURES = ROOT / 'features'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
K_VALUES = tuple(range(2, 9))
SEEDS = (0, 1, 2, 3, 4)
N_INIT = 10
MAX_ITER = 300
SILHOUETTE_SAMPLE_SIZE = 5000
SAMPLE_SEED = 3407
GMM_COVARIANCE = 'diag'
GMM_N_INIT = 3
GMM_MAX_ITER = 300
FINAL_K = 2  # User confirmed K=2 after reviewing the selection and robustness results.

def ensure_dirs():
    for path in (FEATURES, RESULTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
