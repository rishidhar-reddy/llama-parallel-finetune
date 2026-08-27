"""Convert a PyTorch state dict from DDP format to tensor‑parallel (TP) shards.

This script reads a checkpoint saved by `torch.save(model.state_dict())` (for
example, LoRA weights trained via DDP) and splits the matrices across
`num_partitions` shards.  Weight tensors with dimension ≥2 are partitioned
along the first dimension; smaller tensors are replicated across shards.  The
resulting shards are saved under `<output_dir>/tp_rank{rank}_{basename}`.

The conversion only operates on the weight tensor shapes and does not change
the underlying model architecture.  It is suitable for demonstrating how one
might distribute adapter weights for tensor parallel inference.

Each shard carries a reserved ``TP_META_KEY`` entry recording which keys were
sharded and how many partitions were used, so ``tp_to_ddp.merge_shards`` can
reassemble the checkpoint without having to guess. Guessing is not safe: a
sharded tensor whose pieces happen to be identical -- LoRA ``B`` matrices are
zero-initialised, so this is the common case, not a corner case -- is
indistinguishable from a replicated one by inspection alone.
"""

import argparse
import os
from typing import Dict, List

import torch


TP_META_KEY = "__tp_meta__"


def split_state_dict(state_dict: Dict[str, torch.Tensor], num_partitions: int) -> List[Dict[str, torch.Tensor]]:
    """Split weight matrices across the first dimension into `num_partitions` shards."""
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

    shards: List[Dict[str, torch.Tensor]] = [{} for _ in range(num_partitions)]
    split_keys: List[str] = []

    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and tensor.shape[0] >= num_partitions:
            # tensor_split, not chunk: chunk returns *at most* num_partitions
            # pieces and silently returns fewer when the rows do not divide
            # evenly (4 rows over 3 partitions yields 2 chunks), which then
            # raises IndexError below.
            pieces = torch.tensor_split(tensor, num_partitions, dim=0)
            for i in range(num_partitions):
                shards[i][name] = pieces[i].clone()
            split_keys.append(name)
        else:
            # replicate scalar/bias and 1D tensors to all shards
            for i in range(num_partitions):
                shards[i][name] = tensor.clone()

    for shard in shards:
        shard[TP_META_KEY] = {
            "version": 1,
            "num_partitions": num_partitions,
            "split_keys": sorted(split_keys),
        }
    return shards


def main():
    parser = argparse.ArgumentParser(description="Convert DDP checkpoint to TP shards")
    parser.add_argument("--input", type=str, required=True, help="Path to input .bin or .pt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write TP shards")
    parser.add_argument("--num_partitions", type=int, default=2, help="Number of TP partitions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    state_dict = torch.load(args.input, map_location="cpu")
    shards = split_state_dict(state_dict, args.num_partitions)

    base = os.path.basename(args.input)
    for i, shard in enumerate(shards):
        shard_path = os.path.join(args.output_dir, f"tp_rank{i}_{base}")
        torch.save(shard, shard_path)
        print(f"Saved shard {i} to {shard_path}")


if __name__ == "__main__":
    main()
