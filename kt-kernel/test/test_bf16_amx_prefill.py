"""Check BF16 AMX prefill against the AVX512 matvec path."""

import os

import pytest
import torch

import amx_backend


def _run(backend, expert_counts):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_BF16_PREFILL_AMX"] = backend

    torch.manual_seed(13)
    experts, hidden, intermediate, qlen = 2, 1024, 1024, 64
    gate = (torch.randn(experts, intermediate, hidden, dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    up = (torch.randn(experts, intermediate, hidden, dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    down = (torch.randn(experts, hidden, intermediate, dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    pool, _, moe = amx_backend.build_moe("AMXBF16_MOE", experts, hidden, intermediate, qlen, (gate, up, down))

    ids, weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    result = {
        "prefill": amx_backend.forward(pool, moe, ids, weights, inputs, qlen),
        # One-token decode keeps using the matvec path regardless of the gate.
        "decode": amx_backend.forward(pool, moe, ids, weights, inputs, 1),
    }
    reference = amx_backend.swiglu_reference(inputs, ids, weights, gate.float(), up.float(), down.float())
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_bf16_amx_prefill_matches_avx512(tmp_path, expert_counts):
    amx_backend.require_amx("AMXBF16_MOE")
    (fallback, reference), (actual, _) = amx_backend.run_both(__name__, tmp_path, expert_counts=expert_counts)

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)
    amx_backend.assert_matches_reference(actual, reference)
