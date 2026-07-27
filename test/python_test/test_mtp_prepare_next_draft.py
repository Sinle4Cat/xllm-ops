#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
# ==============================================================================

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def mtp_prepare_next_draft_golden(
    accepted_tokens,
    accepted_embeddings,
    embedding_placeholder,
    base_positions,
    base_kv_seq_lens,
    block_tables,
    block_size,
):
    batch_size, speculative_width = accepted_tokens.shape
    accepted_lengths = accepted_tokens.ge(0).sum(dim=1).to(torch.long)
    rows = torch.arange(batch_size, dtype=torch.long)
    last_indices = (accepted_lengths - 1).clamp_min(0)
    previous_indices = (accepted_lengths - 2).clamp_min(0)

    last_tokens = accepted_tokens[rows, last_indices]
    gathered_previous_tokens = accepted_tokens[rows, previous_indices]
    has_previous = accepted_lengths.gt(1)
    previous_tokens = torch.where(
        has_previous, gathered_previous_tokens, last_tokens)
    draft_token_ids = torch.stack(
        (previous_tokens, last_tokens), dim=1).flatten().to(torch.int32)

    last_embeddings = accepted_embeddings[rows, last_indices]
    gathered_previous_embeddings = accepted_embeddings[rows, previous_indices]
    previous_embeddings = torch.where(
        has_previous.unsqueeze(1),
        gathered_previous_embeddings,
        embedding_placeholder.unsqueeze(0).expand(batch_size, -1),
    )
    draft_embeddings = torch.stack(
        (previous_embeddings, last_embeddings), dim=1).flatten(0, 1)

    draft_base_positions = (
        base_positions[:batch_size].to(torch.long) + accepted_lengths)
    draft_positions = torch.stack(
        (draft_base_positions - 1, draft_base_positions), dim=1
    ).flatten().to(torch.int32)
    draft_kv_seq_lens = (
        base_kv_seq_lens[:batch_size].to(torch.long) + accepted_lengths
    ).to(torch.int32)

    repair_positions = torch.where(
        accepted_lengths.eq(speculative_width),
        draft_base_positions - 1,
        draft_base_positions + 1,
    )
    cache_positions = torch.stack(
        (repair_positions, draft_base_positions), dim=1)
    block_indices = torch.div(
        cache_positions, block_size, rounding_mode="floor")
    valid = cache_positions.ge(0) & block_indices.ge(0)
    valid &= block_indices.lt(block_tables.size(1))
    safe_block_indices = block_indices.clamp(0, block_tables.size(1) - 1)
    block_ids = block_tables[rows.unsqueeze(1), safe_block_indices]
    valid &= block_ids.ge(0)
    draft_cache_slots = torch.where(
        valid,
        block_ids.to(torch.long) * block_size + cache_positions.remainder(block_size),
        torch.zeros_like(cache_positions),
    ).flatten().to(torch.int32)

    return (
        draft_token_ids,
        draft_embeddings,
        draft_positions,
        draft_kv_seq_lens,
        draft_cache_slots,
    )


def run_and_check(
    accepted_tokens,
    accepted_embeddings,
    embedding_placeholder,
    base_positions,
    base_kv_seq_lens,
    block_tables,
    block_size,
):
    expected = mtp_prepare_next_draft_golden(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_tables,
        block_size,
    )
    actual = custom_ops.mtp_prepare_next_draft_npu(
        accepted_tokens.npu(),
        accepted_embeddings.npu(),
        embedding_placeholder.npu(),
        base_positions.npu(),
        base_kv_seq_lens.npu(),
        block_tables.npu(),
        block_size,
    )
    torch.npu.synchronize()

    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.dtype == expected_tensor.dtype
        assert tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
        assert torch.equal(actual_tensor.cpu(), expected_tensor)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mtp_prepare_next_draft_mixed_acceptance(dtype):
    batch_size = 4
    speculative_width = 4
    hidden_size = 16
    block_size = 4
    accepted_tokens = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, -1],
            [30, 31, -1, -1],
            [40, -1, -1, -1],
        ],
        dtype=torch.int64,
    )
    accepted_embeddings = torch.arange(
        batch_size * speculative_width * hidden_size, dtype=torch.float32
    ).reshape(batch_size, speculative_width, hidden_size).to(dtype)
    embedding_placeholder = torch.full(
        (hidden_size,), -7.0, dtype=dtype)
    base_positions = torch.tensor([3, 7, 11, 15], dtype=torch.int32)
    base_kv_seq_lens = torch.tensor([4, 8, 12, 16], dtype=torch.int32)
    block_tables = torch.arange(40, 64, dtype=torch.int32).reshape(4, 6)

    run_and_check(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_tables,
        block_size,
    )


def test_mtp_prepare_next_draft_invalid_cache_locations():
    hidden_size = 16
    accepted_tokens = torch.tensor(
        [[10, -1], [20, 21], [30, -1]], dtype=torch.int64)
    accepted_embeddings = torch.arange(
        3 * 2 * hidden_size, dtype=torch.float32
    ).reshape(3, 2, hidden_size).to(torch.float16)
    embedding_placeholder = torch.full(
        (hidden_size,), -3.0, dtype=torch.float16)
    base_positions = torch.tensor([-2, 20, 1], dtype=torch.int32)
    base_kv_seq_lens = torch.tensor([0, 21, 2], dtype=torch.int32)
    block_tables = torch.tensor(
        [[1, 2], [3, 4], [-1, 5]], dtype=torch.int32)

    run_and_check(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_tables,
        block_size=4,
    )
