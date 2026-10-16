"""Compare old/new expert updates on identical CUDA features, without detector noise."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
import torch
from torch import nn
from torch.nn import functional as F
from nets.maddg import Discriminator1
from utils.latent_adversarial import Adversary, weighted_generator
import latent_dg_config as settings


class FixedReference(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.register_buffer('reference', reference)

    def forward(self, images):
        return None, None, self.reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/latent_ablation_validation/expert_update_parity.json')
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = settings.resolve('hard_expert')
    migrated = Adversary.__new__(Adversary)
    migrated.config = cfg
    migrated.discriminators = [Discriminator1().cuda() for _ in range(2)]
    legacy = deepcopy(migrated.discriminators)
    migrated.optimizers = [torch.optim.Adam(d.parameters(), lr=1e-5) for d in migrated.discriminators]
    legacy_opts = [torch.optim.Adam(d.parameters(), lr=1e-5) for d in legacy]
    labels = torch.tensor([0, 1, 0, 1], device='cuda')
    q = F.one_hot(labels, 2).float()
    migrated.experts = [FixedReference(torch.randn(2,512,20,20,device='cuda')) for _ in range(2)]
    max_gradient, max_state = 0., 0.
    for step in range(2):
        features = torch.randn(4,512,20,20,device='cuda',requires_grad=True)
        before = features.sum()*0
        for k,d in enumerate(legacy):
            d.eval().requires_grad_(False)
            before = before + (d(features[labels==k])-1).square().flatten(1).mean(1).sum()/4
        after = weighted_generator(migrated.discriminators,features,q)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        a = torch.autograd.grad(before,features,retain_graph=True)[0]
        b = torch.autograd.grad(after,features)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        max_gradient = max(max_gradient,float((a-b).abs().max()))
        for k,(d,opt) in enumerate(zip(legacy,legacy_opts)):
            d.train().requires_grad_(True); opt.zero_grad(set_to_none=True)
            value = .5*((d(migrated.experts[k].reference)-1).square().mean()+d(features[labels==k].detach()).square().mean())
            value.backward(); opt.step()
        migrated.update(features,features,labels,q)
        for a,b in zip(legacy,migrated.discriminators):
            for key,value in a.state_dict().items():
                other=b.state_dict()[key]
                torch.testing.assert_close(value,other,rtol=0,atol=0)
                max_state=max(max_state,float((value.double()-other.double()).abs().max()))
    report=dict(passed=True,steps=2,identical_features=True,max_generator_gradient_difference=max_gradient,
                max_discriminator_state_difference=max_state,cudnn_benchmark=False)
    output=args.output
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
