# Llama Parallel Fine-Tuning — Engineering Deep Dive

> A component-level walkthrough of how this repository trains Llama-style models
> across GPUs with parameter-efficient adapters. Traced to specific files and lines.

---

## 1. What this repository is

A working study of the two axes you can scale fine-tuning along, with the plumbing to
move a checkpoint between them:

- **Data parallelism (DDP)** — replicate the model, shard the batch.
- **Tensor parallelism (TP)** — shard a single layer's weights across devices.
- **Parameter-efficient adapters** — LoRA and AdaLoRA via `peft`, so only a small
  fraction of parameters carry gradients.

Target benchmark is GLUE. It is not a framework; it is a set of readable reference
implementations plus benchmarks, which is the more useful thing to read.

---

## 2. Layout

| Path | Lines | Role |
|---|---|---|
| `train_ddp_lora.py` | 298 | DDP + LoRA training loop |
| `train_ddp_adalora.py` | 273 | Same, with AdaLoRA's rank allocation |
| `tp_demo/tensor_parallel_linear.py` | 50 | Column-parallel linear layer |
| `utils/ddp_to_tp.py` | ~85 | Shard a checkpoint across TP ranks |
| `utils/tp_to_ddp.py` | ~110 | Merge TP shards back |
| `profiling/benchmark_tp_vs_ddp.py` | 110 | Latency comparison |
| `utils/glue_dataloader.py` | 100 | GLUE task loading and tokenization |
| `utils/metrics.py` | 27 | Accuracy; Pearson/Spearman for STS-B |
| `utils/hooks.py` | 79 | Forward-hook helpers for activation capture |
| `utils/s3_backup.py` | 64 | Periodic checkpoint upload |
| `tests/test_checkpoint_conversion.py` | — | 34 tests over the shard/merge round trip |

---

## 3. The DDP training loop

`train_ddp_lora.py` is the reference path. What it gets right:

- **`setup_distributed()` (line 79)** reads the torchrun-provided environment and calls
  `dist.init_process_group(backend="nccl")`. NCCL means this path is CUDA-only —
  see *Known gaps*.
- **Gradient accumulation** decouples the optimizer step from `--per_device_batch_size`,
  so effective batch size survives a smaller GPU.
- **Mixed precision** behind `--fp16`, using autocast plus a gradient scaler.
- **Warmup** via `--warmup_ratio` (default 0.06), the standard GLUE recipe.
- **Rank-0-only evaluation.** Only one process runs validation and writes checkpoints,
  which avoids the classic bug of every rank racing to write the same file.
- **Optional S3 backup** on an interval (`--s3_interval`, 0 disables).

LoRA is configured through `peft.LoraConfig` with `--lora_r` (8), `--lora_alpha` (16),
`--lora_dropout` (0.05) and a `--lora_target_modules` list. `train_ddp_adalora.py` is
the same loop with `AdaLoraConfig`, which redistributes rank budget across modules
during training rather than fixing it up front.

---

## 4. Tensor parallelism

`tp_demo/tensor_parallel_linear.py` implements a **column-parallel** linear layer:
the output dimension is partitioned across devices, each holds its own slice of the
weight, and the partial outputs are concatenated on the last dimension.

`_partition_sizes` (line 31) distributes the remainder one row at a time, so an output
dimension that does not divide evenly still uses every device.

The module is explicit that it is educational: the input is broadcast and the outputs
gathered with plain `.to(device)` calls, with no overlap of communication and
computation. That honesty is worth preserving — real TP implementations
(Megatron-style) fuse the all-gather into the backward pass.

---

## 5. Checkpoint conversion — the subtle part

Moving a checkpoint between DDP and TP layouts is where the real bugs live, because
nothing here raises when it goes wrong. You get a file that loads fine and is quietly
incomplete.

`utils/ddp_to_tp.py` shards tensors of rank ≥2 along dimension 0 and replicates
everything else. `utils/tp_to_ddp.py` reverses it.

Three defects were found and fixed by writing the round-trip tests:

1. **`torch.chunk` does not return the number of pieces you ask for.** It returns *at
   most* that many. Four rows over three partitions gives two chunks, so indexing the
   third raised `IndexError`. Now uses `torch.tensor_split`, which always returns
   exactly `num_partitions`.

2. **The merge concatenated only when every shard had an equal first dimension** —
   exactly backwards, since an uneven split is precisely what produces unequal shards.
   Those fell through to `tensors[0]`, so merging a 7-row tensor sharded 4 ways
   returned 2 rows and silently discarded the other 5.

3. **Shard files were sorted lexicographically**, placing `tp_rank10` before
   `tp_rank1`. At ten or more partitions the checkpoint reassembled with its rows
   scrambled. `sort_shard_files` now orders by parsed rank.

Fixing (2) correctly required knowing which keys were sharded, and that cannot be
inferred safely: a sharded tensor whose pieces are identical is indistinguishable from
a replicated one, and **LoRA `B` matrices are zero-initialised**, so that is the common
case rather than a corner case. Each shard now carries a reserved `__tp_meta__` entry
recording the sharded keys and partition count. Legacy shards fall back to inspection
and emit a `RuntimeWarning`.

---

## 6. Benchmarking

`profiling/benchmark_tp_vs_ddp.py` measures forward-pass latency for a linear
projection under both strategies, reporting **p50/p95/p99** rather than a mean — the
right call, since tail latency is what distributed training actually suffers from. It
calls `torch.cuda.synchronize()` before and after each timed region, which is the
detail most naive GPU benchmarks get wrong.

`profiling/profiler_trace.py` wraps the PyTorch profiler for per-op traces.

---

## 7. Known gaps

| Gap | Detail |
|---|---|
| **CUDA-only training** | `init_process_group(backend="nccl")` is hardcoded. There is no `gloo` fallback, so neither training script runs on CPU or Apple Silicon. The checkpoint utilities and their tests do run anywhere. |
| **TP is a demo, not a system** | The parallel layer covers a single `nn.Linear`. It is not wired into the Llama model itself, so TP and LoRA training are not actually combined. |
| **No CI** | 34 tests exist and pass, but nothing runs them automatically. |
| **Benchmarks need GPUs** | `benchmark_tp_vs_ddp.py` calls `torch.cuda.synchronize()` unguarded. |
| **No LICENSE** | Neither a file nor a declaration. |

### What I would build next

1. A `gloo` fallback so a single-process CPU smoke test can run the training loop.
2. Wire `TensorParallelLinear` into an actual attention projection.
3. GitHub Actions running the checkpoint tests — they need no GPU.
4. Record benchmark results into `results/` as committed JSON rather than prose.

---

## 8. Attribution

Original implementation by **aditya-dawadikar** across 11 commits, last pushed
2025-11-24. Full history preserved in this repository.

The checkpoint-conversion fixes, the 34-test round-trip suite, and this document are
my own work.
