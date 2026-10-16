"""Numerical/gradient contracts for the new offline ablation (CPU)."""
from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import latent_dg_config as settings
from utils.latent_membership import assignment
from utils.latent_adversarial import Adversary, weighted_generator, soft_cross_entropy, KWayDiscriminator


class TinyExpert(nn.Module):
    def forward(self, images):
        return None, None, images * 0.7


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_onehot_loss_and_gradient_match_historical(self):
        features = torch.randn(5, 3, 4, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 1, 0, 1])
        critics = [nn.Conv2d(3, 1, 1), nn.Conv2d(3, 1, 1)]
        legacy = features.sum() * 0
        for k, d in enumerate(critics):
            d.eval().requires_grad_(False)
            legacy = legacy + (d(features[labels == k]) - 1).square().flatten(1).mean(1).sum() / len(features)
        actual = weighted_generator(critics, features, F.one_hot(labels, 2).float())
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        a = torch.autograd.grad(actual, features, retain_graph=True)[0]
        b = torch.autograd.grad(legacy, features)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_soft_routing_uses_both_branches(self):
        features = torch.randn(3, 2, 2, 2, requires_grad=True)
        critics = [nn.Conv2d(2, 1, 1), nn.Conv2d(2, 1, 1)]
        q = torch.tensor([[.7,.3],[.4,.6],[.2,.8]], requires_grad=True)
        value = weighted_generator(critics, features, q)
        reference = sum(((d(features)-1).square().flatten(1).mean(1)*q[:,k].detach()).sum() for k,d in enumerate(critics))/3
        torch.testing.assert_close(value, reference)
        value.backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNone(q.grad)
        self.assertTrue(all(p.grad is None for d in critics for p in d.parameters()))

    def test_ce_hard_equivalence_and_prior_stationary_point(self):
        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        torch.testing.assert_close(soft_cross_entropy(logits, F.one_hot(labels, 2).float()), F.cross_entropy(logits, labels))
        prior = torch.tensor([[.46,.54]]).expand(4,-1)
        optimal = prior.log().clone().requires_grad_(True)
        gradient = torch.autograd.grad(soft_cross_entropy(optimal, prior), optimal)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)

    def test_discriminator_update_detaches_features_and_skips_missing_reference(self):
        a = Adversary.__new__(Adversary)
        a.config = settings.resolve('soft_expert', 'offline100')
        a.experts = [TinyExpert(), TinyExpert()]
        a.discriminators = [nn.Conv2d(3,1,1), nn.Conv2d(3,1,1)]
        a.optimizers = [torch.optim.SGD(d.parameters(), lr=.01) for d in a.discriminators]
        features = torch.randn(4,3,4,4, requires_grad=True)
        q = torch.tensor([[.7,.3]] * 4, requires_grad=True)
        before = deepcopy(a.discriminators[1].state_dict())
        loss, updates, _ = a.update(torch.randn_like(features), features, torch.zeros(4,dtype=torch.long), q)
        self.assertTrue(np.isfinite(loss)); self.assertEqual(updates, [1,0])
        self.assertIsNone(features.grad); self.assertIsNone(q.grad)
        for name, value in before.items():
            torch.testing.assert_close(value, a.discriminators[1].state_dict()[name], rtol=0, atol=0)

    def test_kway_general_k_and_gradient_isolation(self):
        d = KWayDiscriminator(3, input_channels=4).eval()
        self.assertEqual(tuple(d(torch.randn(2,4,20,20)).shape), (2,3))
        a = Adversary.__new__(Adversary)
        a.config = settings.resolve('soft_kway', 'offline100')
        a.prior = torch.tensor([.48,.52])
        a.discriminators = [nn.Sequential(nn.Flatten(), nn.Linear(12,2))]
        a.optimizers = [torch.optim.SGD(a.discriminators[0].parameters(), lr=.01)]
        features = torch.randn(3,3,2,2, requires_grad=True)
        q = torch.tensor([[.7,.3],[.4,.6],[.2,.8]])
        a.generator_loss(features,q).backward()
        self.assertGreater(features.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in a.discriminators[0].parameters()))
        features.grad = None
        a.update(features,features,q.argmax(1),q)
        self.assertIsNone(features.grad)
        self.assertTrue(any(p.grad is not None for p in a.discriminators[0].parameters()))

    def test_assignment_and_config_validation(self):
        x = np.array([[0.,0.],[1.,2.],[5.,5.]])
        centers = np.array([[0.,1.],[4.,4.]])
        q = assignment(x,centers,1.)
        np.testing.assert_allclose(q.sum(1),1)
        np.testing.assert_array_equal(q.argmax(1), ((x[:,None]-centers)**2).sum(2).argmin(1))
        for name in settings.EXPERIMENTS:
            self.assertEqual(settings.resolve(name)['experiment'],name)
        invalid = settings.resolve('hard_kway'); invalid['assignment']='soft'
        with self.assertRaises(ValueError): settings.validate(invalid)


if __name__ == '__main__':
    unittest.main()
