# Copyright 2026 Individual Contributor: LionG1001
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent dense/analytic references; no model or accelerator is required."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[3] / "verl/trainer/distillation/full_vocab.py"
# Permit the numeric tests without importing optional Ray/engine dependencies.
spec = importlib.util.spec_from_file_location("verl_full_vocab_numeric_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
loss_fn = module.full_vocab_reverse_kl


class FullVocabReverseKLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.student = torch.randn(2, 7, 19, requires_grad=True)
        self.teacher = torch.randn(2, 7, 19, requires_grad=True)
        self.mask = torch.tensor(
            [[False, False, True, True, True, False, False], [False, False, False, True, True, True, True]]
        )

    def test_dense_fp64_and_analytic_student_gradient(self):
        result = loss_fn(self.student, self.teacher, self.mask, token_chunk_size=2)
        log_s = self.student.detach().double().log_softmax(-1)
        log_t = self.teacher.detach().double().log_softmax(-1)
        p = log_s.exp()
        kl = (p * (log_s - log_t)).sum(-1)
        shifted = torch.cat([self.mask[:, 1:], torch.zeros(2, 1, dtype=torch.bool)], 1)
        counts = shifted.sum(-1)
        expected = (kl * shifted).sum(-1) / counts
        torch.testing.assert_close(result.sequence_losses.double(), expected, atol=2e-7, rtol=2e-6)
        result.sequence_losses.mean().backward()
        analytic = p * (log_s - log_t - kl.unsqueeze(-1))
        analytic *= shifted.unsqueeze(-1) / counts[:, None, None] / 2
        torch.testing.assert_close(self.student.grad.double(), analytic, atol=2e-8, rtol=2e-5)
        self.assertIsNone(self.teacher.grad)

    def test_zero_control_and_chunk_invariance(self):
        same = loss_fn(self.student, self.student.detach(), self.mask)
        torch.testing.assert_close(same.sequence_losses, torch.zeros(2), atol=1e-7, rtol=0)
        a = loss_fn(self.student, self.teacher, self.mask, token_chunk_size=1)
        b = loss_fn(self.student, self.teacher, self.mask, token_chunk_size=17)
        torch.testing.assert_close(a.sequence_losses, b.sequence_losses, atol=2e-7, rtol=2e-6)
        ga = torch.autograd.grad(a.sequence_losses.sum(), self.student, retain_graph=True)[0]
        gb = torch.autograd.grad(b.sequence_losses.sum(), self.student)[0]
        torch.testing.assert_close(ga, gb, atol=2e-8, rtol=2e-6)

    def test_unsampled_class_affects_full_distribution(self):
        changed = self.teacher.detach().clone()
        changed[..., -1] += 4
        a = loss_fn(self.student, self.teacher, self.mask)
        b = loss_fn(self.student, changed, self.mask)
        self.assertEqual(a.vocabulary_size, 19)
        self.assertGreater(float((a.sequence_losses - b.sequence_losses).detach().abs().max()), 0.01)

    def test_reject_vocabulary_mismatch_and_bad_causal_masks(self):
        with self.assertRaises(ValueError):
            loss_fn(self.student, self.teacher[..., :-1], self.mask)
        with self.assertRaises(ValueError):
            loss_fn(self.student, self.teacher, torch.zeros_like(self.mask))
        bad = self.mask.clone()
        bad[:, 0] = True
        with self.assertRaises(ValueError):
            loss_fn(self.student, self.teacher, bad)
        with self.assertRaises(ValueError):
            loss_fn(self.student, self.teacher, self.mask, temperature=float("nan"))


if __name__ == "__main__":
    unittest.main()
