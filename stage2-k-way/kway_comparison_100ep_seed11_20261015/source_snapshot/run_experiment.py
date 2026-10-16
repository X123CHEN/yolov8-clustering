"""Run archived experiment modules against the project data and expert weights."""
from pathlib import Path
import runpy
import sys

PROJECT = Path('/data4/czh/yolov8-cluster/yolov8-hard-clustering')
SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))
import utils.comparison_training as training
training.ROOT = PROJECT
training.EXPERT_ROOT = PROJECT.parent/'yolov8-pretrain/results/expert_pretrain_20261004_200ep'
import utils.comparison_evaluation as evaluation
evaluation.ROOT = PROJECT
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name='__main__')
