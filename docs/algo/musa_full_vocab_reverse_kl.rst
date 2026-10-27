Full-vocabulary reverse KL on MUSA (experimental fork)
====================================================

Last updated: 10/27/2026.

This change adds a direct dense reverse-KL loss and a native FSDP integration
point. It is not a replacement of the existing PPO/Ray recipe, and it does not
claim complete DeepSeek-V4-Flash or production SGLang support.

Algorithm and reduction
-----------------------

At each Student-generated prefix, the loss sums
``P_student(v) * (log P_student(v) - log P_teacher(v))`` over the complete output
vocabulary. Both distributions are normalized over all classes. Teacher logits
are detached, and the sampled trajectory is treated as fixed for this update.
This is direct distribution-loss backpropagation, not a sampled-token PG loss.

``verl/trainer/distillation/full_vocab.py`` accepts unshifted [B,S,V] logits
and a boolean response mask. Logits at position t-1 predict token t, so the
mask is shifted once. Prompt and padding are excluded; the first response EOS
is included by the caller. Each sequence is normalized by its response count,
and callers explicitly choose the global sample reduction. The current recipe
averages32 per-sequence means across8 ranks with4 local microbatches.

Chunks cover response positions, never a selected vocabulary subset. The code
still materializes full logits and autograd state: chunking is not a claim of
constant-memory loss or vocabulary parallelism. Empty-response sequences,
different vocabulary geometry and invalid causal masks are rejected.

Native engine integration
-------------------------

``verl/workers/engine/fsdp/full_vocab_impl.py`` defines
``FullVocabReverseKLMixin``. It reads dense model inputs, calls an explicitly
frozen local Teacher under no_grad, and returns differentiable sequence losses
to the caller. Optional evidence hooks observe actual tensors. Native FSDP
retains responsibility for backward, synchronization, AdamW, scheduling and
checkpointing. This mixin does not implement a Ray/main_ppo OPD driver.

The MUSA experiment combined this mixin with the official hardware-plugin's
``FSDPMUSAEngineWithLMHead``. Device and MCCL support were reused, not rewritten.
The environment's Torch/MUSA2.7.1 was preserved. Setting
``VERL_DISABLE_FLASH_ATTN_CE=1`` now avoids importing optional CUDA FlashAttention
CE before checking the explicit opt-out. Its default remains unchanged.

Validation and limitations
--------------------------

.. code-block:: bash

   python -m unittest tests/trainer/distillation/test_full_vocab_reverse_kl.py

Four independent CPU cases compare dense FP64 and analytic gradients, zero and
chunk invariance, the effect of an unsampled vocabulary class, and invalid
geometry/masks. They are mathematical references, not CPU model execution.

The historical native-engine project driver completed150 updates on8 MUSA
GPUs with Qwen2.5-1.5B-Instruct Student and frozen3B-Instruct Teacher. Full
FP32/TF32off, full151936 classes, batch32/accumulation4, and native HF KV
generation were used. It processed4800 independent samples and1,495,748
response tokens; all4800 probability batches passed, maximum logprob
difference0.00041771. Native checkpoints3/50/100/150 and final HF export were
saved. Actual HF-loaded parameters matched the final native Student hash.

On fixed GSM8K test1319, Teacher scored1141, initial Student966, trained
Student954. No net task gain was established. The complete-cycle median was
21.879 seconds including online checks and save cycles; this is not a
production throughput benchmark or a universal framework comparison.

The real-backward Tensor hook used by the project driver must not be replaced
by an extra autograd traversal over FSDP-managed outputs: an earlier diagnostic
did that and failed before any optimizer update. GPU evidence applies to the
frozen experiment source; packaging/lint changes have separate CPU checks.

Bridge dependency scope
-----------------------

The validated FSDP path does not use NVIDIA Megatron-Bridge. The current
Megatron engine does: its use_mbridge field is asserted true and resolves to
``megatron.bridge.AutoBridge``; the megatron extra pins Bridge0.6.2. It must not
be confused with a different project named mbridge. Historical NeMo Bridge
patches are outside this PR and are not assumed compatible with that version.

Review full normalization, causal alignment, explicit reduction, frozen Teacher
and actual backward semantics. Packed sequences, CP/TP loss, production Ray
integration, optimized CUDA/MUSA kernels and Flash weights need separate work.

Repository checks and related work
----------------------------------

All four CPU numeric tests and repository-wide pre-commit pass on this
snapshot. Hydra was supplied in an isolated lint tool environment for the
configuration-generation check. No training dependency stack was replaced.
The license checker also recognizes the2026 Individual Contributor header.

Upstream `PR6920 <https://github.com/verl-project/verl/pull/6920>`_ concerns
student-top-K reverse KL; `PR6194 <https://github.com/verl-project/verl/pull/6194>`_
proposes hidden-state transfer and fused full-distribution distillation.
This fork preserves a dense correctness reference and its historical MUSA
experiment for review; it is not a duplicate upstream feature request. Those
open proposals must be reviewed separately before reuse, especially their
parallelism, kernel and hardware assumptions.
