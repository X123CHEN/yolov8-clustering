"""Invoke frozen training/evaluation modules with local data and expert paths."""
import argparse
import importlib
import sys
from pathlib import Path

from inputs import prepare, check

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=PROJECT/'datasets/RDD2022')
    parser.add_argument('--inputs', type=Path, default=PROJECT/'reproductions/relocated_inputs')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('module', nargs='?', choices=[
        'utils.comparison_training', 'new_experiments.kway_domain.training',
        'new_experiments.kway_domain.evaluation', 'new_experiments.kway_domain.verify'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    inputs = prepare(PROJECT, dataset, args.inputs)
    if args.check_data:
        check(PROJECT, dataset, inputs)
        if not args.module:
            return
    if not args.module:
        parser.error('Choose a module or use --check-data')
    frozen = PROJECT/'stage2-k-way/kway_comparison_100ep_seed11_20261015/source_snapshot'
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(frozen))

    class LocalRoot(type(PROJECT)):
        def __truediv__(self, key):
            key = str(key)
            aliases = {
                'datasets/RDD2022/ImageSets': inputs/'country',
                'latent_domain_analysis/domain_splits': inputs/'latent',
                'latent_domain_analysis/results/final_cluster_labels.csv': inputs/'final_cluster_labels.csv',
            }
            if key in aliases:
                return aliases[key]
            prefix = 'datasets/RDD2022'
            if key == prefix or key.startswith(prefix+'/'):
                return dataset/key[len(prefix):].lstrip('/')
            return PROJECT/key

    import utils.comparison_training as training
    training.ROOT = LocalRoot(PROJECT)
    training.EXPERT_ROOT = PROJECT/'snapshot_dependencies/experts'
    import utils.comparison_evaluation as evaluation
    evaluation.ROOT = LocalRoot(PROJECT)
    sys.argv = [args.module] + args.arguments
    importlib.import_module(args.module).main()


if __name__ == '__main__':
    main()
