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

"""Dense, direct full-vocabulary GKD extension for native FSDP engines.

The mixin deliberately leaves sharding, gradient synchronization, optimizer
steps and checkpointing to the selected native (including hardware-plugin)
FSDP engine. It does not enter PPO's sampled-token surrogate objective.
"""

import torch

from verl.trainer.distillation.full_vocab import full_vocab_reverse_kl


class FullVocabReverseKLMixin:
    """Attach an already frozen local Teacher and an optional evidence hook.

    Subclasses combine this with their device-specific FSDP language-model
    engine. Inputs are dense [B,S], and ``loss_mask`` identifies response IDs.
    ``sequence_losses`` remains differentiable for explicit caller reduction.
    """

    teacher_model = None
    full_vocab_evidence_hook = None

    def prepare_model_inputs(self, micro_batch):
        """Extract dense causal-LM inputs without altering response masks."""
        inputs = {key: micro_batch[key] for key in ["input_ids", "attention_mask", "position_ids"]}
        return inputs, {}

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        """Score identical prefixes with the frozen Teacher and return sequence means."""
        if self.teacher_model is None or any(p.requires_grad for p in self.teacher_model.parameters()):
            raise ValueError("Full-vocabulary distillation requires a separately frozen Teacher")
        inputs, _ = self.prepare_model_inputs(micro_batch)
        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs, use_cache=False).logits
        objective = full_vocab_reverse_kl(output.logits, teacher_logits, micro_batch["loss_mask"].bool())
        if self.full_vocab_evidence_hook is not None:
            self.full_vocab_evidence_hook(output.logits, teacher_logits, micro_batch, objective)
        return {
            "sequence_losses": objective.sequence_losses,
            "token_counts": objective.token_counts,
            "vocabulary_size": objective.vocabulary_size,
        }
