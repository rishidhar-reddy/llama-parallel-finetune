"""Tests for DDP <-> TP checkpoint conversion.

These utilities shard adapter weights across tensor-parallel ranks and merge
them back. A defect here does not raise in training — it silently produces a
checkpoint with missing or misordered weights, which surfaces much later as
unexplained quality loss. The round-trip property (split then merge returns the
original tensor exactly) is the thing worth pinning down.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ddp_to_tp import split_state_dict  # noqa: E402
from utils.tp_to_ddp import merge_shards, sort_shard_files  # noqa: E402


# Deliberately mixes divisible and non-divisible row counts — the non-divisible
# cases are where both utilities used to fail.
SHAPES = [(6, 3), (8, 4), (7, 4), (10, 4), (9, 2), (4, 3), (5, 3), (2, 2), (100, 8)]


class TestSplitStateDict:
    @pytest.mark.parametrize("rows,parts", SHAPES)
    def test_always_returns_exactly_the_requested_number_of_shards(self, rows, parts):
        sd = {"w": torch.randn(rows, 8)}
        assert len(split_state_dict(sd, parts)) == parts

    @pytest.mark.parametrize("rows,parts", SHAPES)
    def test_shard_rows_sum_to_the_original(self, rows, parts):
        sd = {"w": torch.randn(rows, 8)}
        shards = split_state_dict(sd, parts)
        assert sum(s["w"].shape[0] for s in shards) == rows

    def test_one_dimensional_tensors_are_replicated_not_split(self):
        bias = torch.randn(16)
        shards = split_state_dict({"b": bias}, 4)
        assert len(shards) == 4
        for shard in shards:
            assert torch.equal(shard["b"], bias)

    def test_matrix_shorter_than_the_partition_count_is_replicated(self):
        small = torch.randn(2, 8)
        shards = split_state_dict({"w": small}, 4)
        for shard in shards:
            assert torch.equal(shard["w"], small)

    def test_shards_do_not_alias_the_source_tensor(self):
        sd = {"w": torch.zeros(6, 4)}
        shards = split_state_dict(sd, 3)
        shards[0]["w"] += 1.0
        assert torch.equal(sd["w"], torch.zeros(6, 4))


class TestRoundTripFidelity:
    """split -> merge must return the original tensor exactly."""

    @pytest.mark.parametrize("rows,parts", SHAPES)
    def test_matrix_round_trips_exactly(self, rows, parts):
        original = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
        merged = merge_shards(split_state_dict({"w": original}, parts))["w"]
        assert merged.shape == original.shape
        assert torch.equal(merged, original)

    def test_mixed_state_dict_round_trips(self):
        original = {
            "layer.weight": torch.randn(7, 4),   # non-divisible by 4
            "layer.bias": torch.randn(7),        # 1D, replicated
            "scale": torch.tensor(2.0),          # 0D, replicated
        }
        merged = merge_shards(split_state_dict(original, 4))
        assert set(merged) == set(original)
        for key, value in original.items():
            assert torch.equal(merged[key], value), key

    def test_single_partition_is_identity(self):
        original = {"w": torch.randn(5, 3)}
        merged = merge_shards(split_state_dict(original, 1))
        assert torch.equal(merged["w"], original["w"])


class TestShardFileOrdering:
    """Shards must merge in rank order. Lexicographic sorting places
    tp_rank10 before tp_rank1, which reassembles the checkpoint scrambled."""

    def test_double_digit_ranks_sort_numerically(self):
        files = [f"/ckpt/tp_rank{i}_model.pt" for i in range(12)]
        shuffled = sorted(files)  # lexicographic, deliberately wrong order
        assert sort_shard_files(shuffled) == files

    def test_single_digit_ranks_still_sort_correctly(self):
        files = [f"/ckpt/tp_rank{i}_model.pt" for i in range(4)]
        assert sort_shard_files(list(reversed(files))) == files
