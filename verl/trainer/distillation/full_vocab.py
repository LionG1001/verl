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

"""Direct full-vocabulary reverse KL on fixed student-generated trajectories.

This objective is distinct from teacher-top-k forward KL and from a sampled
token policy-gradient estimator. Chunking is over response positions, never
over a selected vocabulary support. The entire head vocabulary is normalized.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FullVocabReverseKLOutput:
    """Per-sequence means plus sums/counts for explicit caller aggregation."""

    sequence_losses: torch.Tensor
    token_sums: torch.Tensor
    token_counts: torch.Tensor
    vocabulary_size: int


def full_vocab_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_chunk_size: int = 32,
) -> FullVocabReverseKLOutput:
    """Compute KL(Student || frozen Teacher) at causal response positions.

    Args:
        student_logits: Unshifted differentiable logits [B, S, V].
        teacher_logits: Same-prefix logits [B, S, V]; detached unconditionally.
        response_mask: Bool [B, S], marking actual response token IDs including
            the first EOS and excluding prompt/pad. Logits at t-1 predict token t.
        temperature: Positive shared distillation temperature. No T-squared
            rescaling is applied; the current OPD recipe explicitly uses T=1.
        token_chunk_size: Response positions per chunk, retaining all V classes.

    Returns:
        Per-sequence token means. Callers choose sequence or global-token
        aggregation explicitly instead of accidentally averaging rank means.

    Raises:
        ValueError: Geometry, vocabulary, mask, or temperature is incompatible.
    """
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and Teacher must share the complete [B,S,V] output geometry")
    if response_mask.shape != student_logits.shape[:2] or response_mask.dtype != torch.bool:
        raise ValueError("Response mask must be bool [B,S]")
    if student_logits.device != teacher_logits.device or student_logits.device != response_mask.device:
        raise ValueError("Logits and mask must reside on the same device")
    if not temperature > 0 or not math.isfinite(temperature) or token_chunk_size < 1:
        raise ValueError("Temperature and token chunk size must be positive and finite")
    if response_mask[:, 0].any():
        raise ValueError("The first input token cannot have a causal response prediction")
    shifted_mask = response_mask[:, 1:]
    counts = shifted_mask.sum(dim=-1)
    if counts.numel() == 0 or (counts == 0).any():
        raise ValueError("Every sequence must contain at least one real response token")
    frozen_teacher = teacher_logits.detach()
    totals = []
    for index in range(student_logits.shape[0]):
        s = student_logits[index, :-1][shifted_mask[index]].float() / temperature
        t = frozen_teacher[index, :-1][shifted_mask[index]].float() / temperature
        chunks = []
        for start in range(0, s.shape[0], token_chunk_size):
            log_s = F.log_softmax(s[start : start + token_chunk_size], dim=-1)
            log_t = F.log_softmax(t[start : start + token_chunk_size], dim=-1)
            chunks.append((log_s.exp() * (log_s - log_t)).sum())
        totals.append(torch.stack(chunks).sum())
    sums = torch.stack(totals)
    return FullVocabReverseKLOutput(sums / counts, sums, counts, student_logits.shape[-1])
