> **Note:** this repository preserves the original authors' full commit history.
> Architecture walkthrough: [docs/SYSTEM-DEEPDIVE.md](docs/SYSTEM-DEEPDIVE.md)

# Llama-3 Fine‑Tuning with DDP, TP, LoRA & AdaLoRA on GLUE

This repository demonstrates how to fine‑tune [Llama‑3](https://huggingface.co/meta-llama) style models on the [GLUE](https://gluebenchmark.com/) benchmark using **Distributed Data Parallel (DDP)**, **Tensor Parallelism (TP)** and parameter‑efficient adapters such as **LoRA** and **AdaLoRA**.  It contains training scripts, benchmarking utilities, conversion helpers and setup scripts to reproduce the experiments.


## Repository Layout

```
llama_ddp_tp_repo/
├── README.md                      – High level overview and usage instructions
├── train_ddp_lora.py              – DDP training script with LoRA
├── train_ddp_adalora.py           – DDP training script with AdaLoRA
├── tp_demo/                       – Simple tensor‑parallel linear layer and demo
│   ├── tensor_parallel_linear.py
│   └── run_tp.py
├── profiling/                     – Profiling and benchmarking utilities
│   ├── profiler_trace.py          – Example of using PyTorch profiler
│   ├── benchmark_tp_vs_ddp.py     – Throughput and latency comparison script
│   └── latency_utils.py           – Helper functions to compute p95/p99
├── utils/                         – Shared utilities
│   ├── glue_dataloader.py         – Data loading for GLUE tasks
│   ├── hooks.py                   – Activation hooks for adapter placement
│   ├── metrics.py                 – Simple evaluation metrics for GLUE
│   ├── s3_backup.py               – Periodic backup to AWS S3
│   ├── ddp_to_tp.py               – Convert DDP weights to TP layout
│   └── tp_to_ddp.py               – Convert TP weights back to DDP
├── scripts/                       – Shell scripts for setup and orchestration
│   ├── setup_env.sh               – Install Python, CUDA, networking and monitoring deps
│   ├── run_training.sh            – Convenient wrapper to launch training
│   └── backup.sh                  – Example script to trigger S3 backup
├── results/                       – Directory for storing logs and evaluation results
└── weights/                       – Directory for saving adapter checkpoints
```

## Requirements

* **Hardware:** A machine with one or more NVIDIA GPUs and CUDA installed.  Multi‑GPU training requires NCCL‑capable interconnects (e.g. NVLink or PCIe).
* **Software:** Python 3.10+ and the packages listed in `requirements.txt`.  Use the provided `scripts/setup_env.sh` to create a virtual environment and install dependencies including:
  * PyTorch with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu118`)
  * Hugging Face Transformers & Datasets
  * [peft](https://github.com/huggingface/peft) for LoRA/AdaLoRA
  * boto3 for S3 backups
  * fvcore and other profiling utilities

## Quick Start

1. **Install dependencies** (once per machine):

   ```bash
   cd llama_ddp_tp_repo
   bash scripts/setup_env.sh
   ```

2. **Fine‑tune with LoRA on a GLUE task** using DDP.  The example below runs SST‑2 on 4 GPUs.  Adjust `--nproc_per_node` for the number of GPUs and hyper‑parameters as needed.

   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   torchrun --nproc_per_node=4 --master_port=29501 \
       train_ddp_lora.py \
       --model_name meta-llama/Meta-Llama-3-8B-Instruct \
       --task_name sst2 \
       --output_dir runs/llama3_lora_sst2 \
       --per_device_batch_size 8 \
       --epochs 3
   ```

3. **Fine‑tune with AdaLoRA**.  AdaLoRA requires calling `update_and_allocate()` at each step; this is handled inside `train_ddp_adalora.py`.

   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   torchrun --nproc_per_node=4 --master_port=29501 \
       train_ddp_adalora.py \
       --model_name meta-llama/Meta-Llama-3-8B-Instruct \
       --task_name mnli \
       --output_dir runs/llama3_adalora_mnli \
       --per_device_batch_size 4 \
       --epochs 3
   ```

4. **Benchmark TP vs DDP**.  The `profiling/benchmark_tp_vs_ddp.py` script measures throughput and latency on a simple linear workload across multiple GPUs.  It prints p50/p95/p99 latencies and throughput for each strategy.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python profiling/benchmark_tp_vs_ddp.py --hidden_dim 4096 --batch_size 32 --num_steps 100
   ```

5. **Back up checkpoints to S3**.  The utilities in `utils/s3_backup.py` periodically upload the contents of your `--output_dir` to an S3 bucket.  Before running, set environment variables for your AWS credentials and bucket name:

   ```bash
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export S3_BUCKET=my-llama-checkpoints
   python utils/s3_backup.py --source_dir runs/llama3_lora_sst2 --bucket "$S3_BUCKET" --interval 3600
   ```

6. **Convert checkpoints between DDP and TP formats**.  After training with DDP, the LoRA weights can be split across tensor parallel shards via `utils/ddp_to_tp.py`.  To merge TP shards back to a single DDP file, use `utils/tp_to_ddp.py`.  These scripts operate on PyTorch state dictionaries and require specifying the number of TP partitions.

## Notes on LoRA and AdaLoRA

* **LoRA** adds trainable rank‑decomposed matrices to existing weight matrices while keeping the original weights frozen.  This drastically reduces the number of trainable parameters and memory footprint.
* **AdaLoRA** extends LoRA by dynamically reallocating the low‑rank budget across layers during training.  It uses the magnitude of gradients to decide which adapters require larger ranks and calls `update_and_allocate()` at each step to adapt the rank allocation.

## Contribution

Feel free to fork this repository and extend the scripts to support different models, tasks or distributed paradigms.  Pull requests are welcome as long as they do not include proprietary code.  For questions or suggestions, open an issue.