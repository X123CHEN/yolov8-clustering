"""Protect the refinement's objective, gradients, BN buffers, and baseline protocol."""
from copy import deepcopy
import unittest

import torch
from torch import nn

import latent_dg_config as settings
from utils.latent_adversarial import Adversary, adversarial_weight, generator_discriminator


class TinyExpert(nn.Module):
    def forward(self, x):
        return None, None, x*.7


class CountingCritic(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 1, 1)
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(len(x))
        return super().forward(x)


class RefinementContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_batch_bn_has_training_outputs_input_gradients_and_no_buffer_updates(self):
        d = nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU(),nn.Conv2d(4,1,1))
        reference = deepcopy(d).train()
        x = torch.randn(5,3,4,4,requires_grad=True)
        before = deepcopy(d.state_dict())
        with generator_discriminator(d, 'batch'):
            actual = d(x)
        torch.testing.assert_close(actual,reference(x.detach()))
        actual.square().mean().backward()
        self.assertGreater(x.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in d.parameters()))
        self.assertFalse(d[1].training)
        self.assertTrue(d[1].track_running_stats)
        for key,value in before.items():
            torch.testing.assert_close(value,d.state_dict()[key],atol=0,rtol=0)

    def test_soft_generator_with_hard_fake_preserves_reference_batch_populations(self):
        a = Adversary.__new__(Adversary)
        a.config = settings.resolve('soft_expert','refine60')
        a.experts = [TinyExpert(),TinyExpert()]
        a.discriminators = [CountingCritic(),CountingCritic()]
        a.optimizers = [torch.optim.SGD(d.parameters(),lr=.01) for d in a.discriminators]
        x = torch.randn(4,3,4,4,requires_grad=True)
        q = torch.tensor([[.8,.2],[.7,.3],[.3,.7],[.2,.8]],requires_grad=True)
        a.generator_loss(x,q).backward()
        self.assertGreater(x.grad.norm().item(),0)
        self.assertIsNone(q.grad)
        x.grad = None
        _,updates,_ = a.update(x.detach(),x,q.argmax(1),q)
        self.assertEqual(updates,[1,1])
        self.assertTrue(all(d.batch_sizes==[4,2,2] for d in a.discriminators))
        self.assertIsNone(x.grad)
        self.assertIsNone(q.grad)

    def test_warmup_and_historical_default(self):
        c = dict(adv_weight=.7,adv_warmup_epochs=5.)
        self.assertEqual(adversarial_weight(c,0),0.)
        self.assertAlmostEqual(adversarial_weight(c,2,50,100),.35)
        self.assertEqual(adversarial_weight(c,5),.7)
        self.assertEqual(adversarial_weight(c,59),.7)
        self.assertEqual(adversarial_weight(dict(adv_weight=.7),0),.7)

    def test_kway_refinement_changes_only_assignment_target(self):
        a,b = [settings.resolve(n,'refine60') for n in ['hard_kway','soft_kway']]
        self.assertEqual(a['kway_confusion_target'],'hard_prior')
        self.assertEqual(a['epochs'],60)
        self.assertEqual(a['scheduler_epochs'],100)
        for c in [a,b]:
            c.pop('assignment');c.pop('experiment')
        self.assertEqual(a,b)

    def test_historical_profile_and_metric_remain_available(self):
        for n in settings.EXPERIMENTS:
            c = settings.resolve(n,'offline100')
            self.assertEqual(c['epochs'],100)
            self.assertEqual(c['membership']['alpha'],1.)
            self.assertEqual(c['adv_weight'],.7)
            self.assertEqual(c['adv_warmup_epochs'],0.)
            self.assertEqual(c['generator_bn_mode'],'running')
            self.assertEqual(c['expert_fake_assignment'],'follow')
            self.assertEqual(c['validation']['confidence_mode'],'legacy_uncertainty')


if __name__=='__main__': unittest.main()
