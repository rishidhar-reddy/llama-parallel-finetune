"""Merge tensor‑parallel shards back into a single DDP checkpoint.

Given multiple shard files created by `ddp_to_tp.py`, this script reconstructs
the original state dict by concatenating the partitioned tensors along their
first dimension and validating that the remaining keys are identical.  All
shards must have the same keys; scalar and bias tensors are taken from the
first shard.
"""

import argparse
import os
import re
import warnings
from glob import glob
from typing import Any, Dict, List

import torch

try:  # package import (tests, `from utils.tp_to_ddp import ...`)
    from .ddp_to_tp import TP_META_KEY
except ImportError:  # direct script run: `python utils/tp_to_ddp.py`
    from ddp_to_tp import TP_META_KEY


_RANK_RE = re.compile(r"tp_rank(\d+)_")


def sort_shard_files(paths: List[str]) -> List[str]:
    """Order shard paths by their numeric rank.

    Plain lexicographic sorting places ``tp_rank10`` before ``tp_rank1``, which
    reassembles the checkpoint with its rows scrambled. Nothing raises -- the
    merged file is simply wrong.
    """
    def rank_of(path: str) -> int:
        match = _RANK_RE.search(os.path.basename(path))
        if match is None:
            raise ValueError(f"Cannot parse a tp_rank<N>_ prefix from {path!r}")
        return int(match.group(1))

    return sorted(paths, key=rank_of)


def load_shards(input_dir: str) -> List[Dict[str, torch.Tensor]]:
    shard_files = sort_shard_files(glob(os.path.join(input_dir, "tp_rank*")))
    shards = []
    for f in shard_files:
        shards.append(torch.load(f, map_location="cpu"))
    if not shards:
        raise RuntimeError(f"No shard files found in {input_dir}")
    return shards


def _split_keys_from_metadata(shards: List[Dict[str, Any]]) -> set:
    """Which keys were sharded, per the metadata written by ddp_to_tp."""
    meta = shards[0].get(TP_META_KEY)
    if isinstance(meta, dict) and "split_keys" in meta:
        return set(meta["split_keys"])

    # Legacy shards predate the metadata key. Fall back to inspection: a
    # replicated tensor is byte-identical across every shard. This is a guess,
    # and it is wrong for a sharded tensor whose pieces are all equal (a
    # zero-initialised LoRA B matrix, for instance), so say so out loud.
    warnings.warn(
        f"Shards carry no {TP_META_KEY!r}; falling back to inspecting the "
        "tensors. Re-export with the current ddp_to_tp.py to remove the "
        "ambiguity.",
        RuntimeWarning,
        stacklevel=2,
    )
    inferred = set()
    for key, first in shards[0].items():
        if not isinstance(first, torch.Tensor) or first.ndim < 2:
            continue
        others = [shard[key] for shard in shards[1:]]
        if any(o.shape != first.shape or not torch.equal(o, first) for o in others):
            inferred.add(key)
    return inferred


def merge_shards(shards: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not shards:
        raise ValueError("No shards to merge")

    split_keys = _split_keys_from_metadata(shards)

    merged: Dict[str, torch.Tensor] = {}
    for key in shards[0]:
        if key == TP_META_KEY:
            continue
        tensors = [shard[key] for shard in shards]
        # Concatenate the keys that were sharded; every other key was
        # replicated, so any shard holds the whole value.
        #
        # The previous heuristic concatenated only when all shards had an equal
        # first dimension, which is exactly backwards: an uneven split produces
        # unequal shards, so those fell through to `tensors[0]` and the merged
        # checkpoint silently kept one shard's rows and dropped the rest.
        merged[key] = torch.cat(tensors, dim=0) if key in split_keys else tensors[0]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge TP shards into a DDP checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing tp_rank*_ files")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged checkpoint")
    args = parser.parse_args()

    shards = load_shards(args.input_dir)
    merged = merge_shards(shards)
    torch.save(merged, args.output)
    print(f"Merged {len(shards)} shards into {args.output}")


if __name__ == "__main__":
    main()
