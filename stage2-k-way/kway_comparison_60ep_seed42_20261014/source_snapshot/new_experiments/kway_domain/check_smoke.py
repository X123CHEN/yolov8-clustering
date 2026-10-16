"""Validate four-mode updates, saved optimizer state and detector-only inference."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from utils.comparison_training import atomic_json, prepare_data
from utils.comparison_evaluation import StrictYOLO
from new_experiments.kway_domain.report import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.suite / 'verification.json').read_text())
    checks = {}
    first = {}
    for name in EXPERIMENTS:
        folder = args.suite / 'checks' / name
        completion = json.loads((folder / 'completion.json').read_text())
        assert completion['smoke_test'] and completion['epochs_completed'] == 1
        state = torch.load(folder / 'training_state.pt', map_location='cpu')
        config = state['config']
        single = 'kway' in name
        assert config['seed'] == 42 and config['epochs'] == 60
        assert config['discriminators'] == (1 if single else len(config['domains']))
        assert bool(config['experts']) != single
        assert state['global_step'] == 3 and state['scheduler']['epochs'] == 60
        assert state['optimizer']['state'] and all(opt['state'] for opt in state['critic_optimizers'])
        audit = json.loads((folder / 'epoch_audit.jsonl').read_text().splitlines()[-1])
        assert audit['images_seen'] == 36 and audit['experts_frozen']
        assert audit['discriminator_updates'] == [3] * config['discriminators']
        grads = audit['gradient_checks']
        assert grads[0]['fab'] > 0 and grads[-1]['domain'] > 0
        first[name] = grads[0]
        checks[name] = dict(steps=3, head_count=config['discriminators'], expert_count=len(config['experts']),
                           peak_memory_gb=audit['peak_mem_GB'], optimizer_states=True,
                           fab_and_domain_gradients=True, checkpoint_complete=True)
        del state
    for baseline, replacement in ((EXPERIMENTS[0],EXPERIMENTS[2]),(EXPERIMENTS[1],EXPERIMENTS[3])):
        for key in ('detection','fab'):
            torch.testing.assert_close(torch.tensor(first[baseline][key]), torch.tensor(first[replacement][key]), rtol=1e-5, atol=1e-6)
    # Source validation image only; no target test inference before final training.
    _,_,_,_,_,_,validation,_ = prepare_data('country_unique')
    model = StrictYOLO(model_path=str(args.suite / 'checks' / EXPERIMENTS[3] / 'best_epoch_weights.pth'),
                       phi='s',input_shape=[640,640],confidence=.001,nms_iou=.5,
                       letterbox_image=False,cuda=True)
    with Image.open(validation[0].split()[0]) as image: predictions = model.predict(image)
    checks['detector_only_inference'] = dict(passed=True, predictions=len(predictions), source_validation_only=True)
    result.update(smoke_passed=True, smoke_checks=checks)
    atomic_json(args.suite / 'verification.json', result)
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
