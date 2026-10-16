"""Meaningful contracts for the structural revision; source-only, no AP tuning."""
from copy import deepcopy
import itertools
import unittest

import numpy as np
import torch
from torch import nn

import latent_dg_config as settings
from utils.latent_adversarial import Adversary, generator_discriminator, soft_cross_entropy
from utils.latent_structure import (ConditionalStyleDiscriminator, replace_batch_norm,
                                   sampled_branch_risk, source_conditions)


class StructuralContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_conditional_priors_use_original_image_counts_and_smooth_missing_patterns(self):
        records = {'a': 'a 0,0,4,4,0 1,1,2,2,0', 'b': 'b 0,0,4,4,1',
                   'c': 'c 0,0,4,4,0 0,0,2,2,1', 'd': 'd 0,0,0,4,0'}
        c, table, info = source_conditions(records, dict(a=0,b=1,c=1,d=0), 2, 2, 2.)
        np.testing.assert_array_equal(c, [1,2,3,0])
        self.assertEqual(sum(map(sum,info['counts'])),4)
        np.testing.assert_allclose(table[1],[2/3,1/3])
        _, empty, _ = source_conditions({'a':records['a'],'b':records['b']},dict(a=0,b=1),2,2,2.)
        np.testing.assert_allclose(empty[3],[.5,.5])
        np.testing.assert_allclose(table.sum(1),1)
        self.assertTrue((table>0).all())

    def test_style_head_independent_of_batch_population_and_prior_stationary(self):
        d = ConditionalStyleDiscriminator(2,channels=3,conditions=4,hidden=8)
        x = torch.randn(3,3,5,5,requires_grad=True)
        c = torch.tensor([0,1,3])
        a = d(x,c)
        for i in range(3):
            torch.testing.assert_close(a[i:i+1],d(x[i:i+1],c[i:i+1]),atol=2e-7,rtol=2e-6)
        q = torch.tensor([[.2,.8],[.6,.4],[.3,.7]],requires_grad=True)
        loss = soft_cross_entropy(a,q)
        with generator_discriminator(d):
            loss = soft_cross_entropy(d(x,c),q)
        loss.backward()
        self.assertGreater(x.grad.norm().item(),0)
        self.assertIsNone(q.grad)
        self.assertTrue(all(p.grad is None for p in d.parameters()))
        logits = q.detach().log().requires_grad_(True)
        torch.testing.assert_close(torch.autograd.grad(soft_cross_entropy(logits,q),logits)[0],
                                   torch.zeros_like(logits),atol=1e-7,rtol=0)

    def test_group_norm_removes_population_and_train_eval_dependence(self):
        d = replace_batch_norm(nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.LeakyReLU(.2)),2)
        self.assertFalse(any(isinstance(m,nn.BatchNorm2d) for m in d.modules()))
        x = torch.randn(4,3,4,4)
        a = d.train()(x)
        torch.testing.assert_close(a,d.eval()(x),atol=0,rtol=0)
        for i in range(4):
            torch.testing.assert_close(a[i:i+1],d(x[i:i+1]),atol=3e-7,rtol=2e-6)

    def test_exact_enumeration_sampled_risk_gradient_matches_full_soft_risk(self):
        # Enumerate all 2**3 assignments: no Monte Carlo tolerance masks a bias.
        q = torch.tensor([[.8,.2],[.3,.7],[.6,.4]],dtype=torch.float64)
        prior = torch.tensor([.55,.45],dtype=torch.float64)
        d = nn.Linear(2,1,bias=False).double()
        real, fake = torch.randn(3,2,dtype=torch.float64), torch.randn(3,2,dtype=torch.float64)
        for k in range(2):
            risk = (((d(real)-1).square()+d(fake).square()).flatten()*q[:,k]).sum()/(6*prior[k])
            full = torch.autograd.grad(risk,d.weight)[0]
            expected = torch.zeros_like(full)
            for choices in itertools.product(range(2),repeat=3):
                mask = torch.tensor(choices)==k
                probability = q[torch.arange(3),torch.tensor(choices)].prod()
                if mask.any():
                    value = sampled_branch_risk(d(real[mask]),d(fake[mask]),3,prior[k])
                    expected += probability*torch.autograd.grad(value,d.weight)[0]
            torch.testing.assert_close(expected,full,atol=1e-12,rtol=1e-12)

    def test_dedicated_routing_rng_roundtrip_and_global_rng_isolation(self):
        a = Adversary.__new__(Adversary)
        a.config = settings.resolve('soft_expert','structural60')
        a.routing_rng = torch.Generator().manual_seed(30011)
        before = torch.get_rng_state().clone()
        state = a.routing_state().clone()
        q = torch.tensor([[.3,.7]]*24)
        first = torch.multinomial(q,1,generator=a.routing_rng)
        a.restore_routing_state(state)
        torch.testing.assert_close(first,torch.multinomial(q,1,generator=a.routing_rng),atol=0,rtol=0)
        torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)
        with self.assertRaises(ValueError):a.restore_routing_state(None)

    def test_structural_pair_and_legacy_defaults_are_separate(self):
        hard,soft = [settings.resolve(n,'structural60') for n in ['hard_kway','soft_kway']]
        for c in [hard,soft]:c.pop('experiment');c.pop('assignment')
        self.assertEqual(hard,soft)
        self.assertEqual(hard['adversarial_feature'],'feat1_style')
        self.assertEqual(hard['coco_period'],5)
        old = settings.resolve('hard_expert','structural60')
        self.assertEqual(old['expert_normalization'],'batch')
        self.assertEqual(old['expert_reference_routing'],'hard')
        self.assertEqual(old['epochs'],100)
        bad = settings.resolve('soft_expert','structural60');bad['expert_normalization']='batch'
        with self.assertRaises(ValueError):settings.validate(bad)


if __name__ == '__main__':unittest.main()
