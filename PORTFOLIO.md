# llama-parallel-finetune — portfolio notes

> **⚠️ Fill in every `[FILL IN]` below and delete this line before sharing.**
> *My contributions* is verifiable from `git log`. *My role on the original build* is
> yours to state accurately — left blank rather than guessed.

---

## What this is

A working study of fine-tuning Llama-style models across GPUs: **distributed data
parallelism (DDP)**, **tensor parallelism (TP)**, and parameter-efficient adapters
(**LoRA** and **AdaLoRA**) benchmarked on GLUE — plus the checkpoint plumbing to move
weights between DDP and TP layouts.

**Stack:** PyTorch · DDP · NCCL · PEFT (LoRA/AdaLoRA) · Hugging Face Transformers · S3

Architecture walkthrough → [`docs/SYSTEM-DEEPDIVE.md`](docs/SYSTEM-DEEPDIVE.md)

Why it is worth talking about: most candidates can describe LoRA. Far fewer can explain
why `torch.chunk` is the wrong primitive for sharding a checkpoint, or why tail latency
is the metric that matters when you benchmark a parallel layer.

---

## Provenance

Original implementation by **aditya-dawadikar**, 11 commits, last pushed 2025-11-24.
This repository preserves that full history; the original is at
[Aditya-Dawadikar/Llama8B-DDP-TP](https://github.com/Aditya-Dawadikar/Llama8B-DDP-TP),
tracked here as `upstream`.

### My role on the original build

[FILL IN — name what you actually worked on. If you did not contribute to the original,
say so; the three bugs found and fixed below stand on their own as a contribution.]

---

## My contributions in this repository

`git log --author="Rishidhar Reddy Garlapati"`

### Found and fixed three correctness bugs in checkpoint conversion (`afd9fb5`)

The DDP↔TP conversion utilities had no test coverage. A defect there does not raise
during training — it produces a checkpoint that loads fine and is quietly incomplete,
surfacing much later as unexplained quality loss. I wrote 34 tests over the round-trip
property, which surfaced three bugs, **two of them silent**:

| # | Bug | Consequence |
|---|---|---|
| 1 | `torch.chunk` returns *at most* n pieces, not exactly n | `IndexError` at 4 rows over 3 partitions |
| 2 | Merge concatenated only when shards had **equal** first dims — backwards | 7-row tensor sharded 4 ways merged back to **2 rows**, 5 silently discarded |
| 3 | Shard files sorted lexicographically | `tp_rank10` before `tp_rank1`; rows scrambled at ≥10 partitions |

Bug 2 could not be fixed by inference alone: a sharded tensor whose pieces are
identical is indistinguishable from a replicated one, and LoRA `B` matrices are
zero-initialised, so that is the *common* case. I added a versioned `__tp_meta__`
record to each shard naming the sharded keys, with a warning-emitting fallback for
legacy checkpoints.

Verified end to end through both CLIs: a 7-row tensor sharded 4 ways now round-trips
exactly. **0 → 34 tests, all passing.**

### Documented the architecture (`<this commit>`)

A component-level deep dive covering the DDP loop, the column-parallel linear layer,
the conversion bugs and why they were silent, and an honest gaps list.

---

## What I would build next

1. A `gloo` backend fallback — `nccl` is hardcoded, so no CPU smoke test is possible.
2. Wire `TensorParallelLinear` into a real attention projection; today TP is a
   standalone demo and is never combined with LoRA training.
3. GitHub Actions for the checkpoint tests — they need no GPU.
4. Commit benchmark results as structured JSON in `results/` instead of prose.
5. A LICENSE file — there is currently none.

---

## Honest limitations

- **Training requires CUDA.** `init_process_group(backend="nccl")` is hardcoded; the
  training scripts will not run on CPU or Apple Silicon. The checkpoint utilities and
  their 34 tests run anywhere.
- **TP is educational.** The layer's own docstring says so — no overlap of
  communication and computation, unlike a Megatron-style implementation.
- **I have not run a full training job.** The bug fixes are verified by unit tests and
  end-to-end CLI round trips, not by a GLUE training run on multiple GPUs.
