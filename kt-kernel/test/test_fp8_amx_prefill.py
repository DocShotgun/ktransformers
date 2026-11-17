"""Check FP8 AMX prefill against the AVX512 matvec path."""

import os

import pytest
import torch

import amx_backend


def dequantize_fp8_blockwise(fp8, scales, group_size=128):
    experts, n, k = fp8.shape
    values = fp8.view(torch.float8_e4m3fn).to(torch.float32)
    values = values.view(experts, n // group_size, group_size, k // group_size, group_size)
    scaled = values * scales.view(experts, n // group_size, 1, k // group_size, 1)
    return scaled.view(experts, n, k)


def _run(backend, expert_counts, hidden=1024):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_FP8_PREFILL_AMX"] = backend

    torch.manual_seed(17)
    experts, intermediate, qlen = 2, 1024, sum(expert_counts)
    group_size = 128

    # Random FP8 codes stress every widening lookup entry (denorms, both signs);
    # kt's decode maps 0x7F/0xFF finite where torch sees NaN, so clamp those
    # codes away and let torch decode the rest exactly like the kernel's LUTs.
    def make_fp8_weights(n, k):
        fp8 = torch.randint(0, 256, (experts, n, k), dtype=torch.int32).to(torch.uint8)
        fp8[(fp8 == 0x7F) | (fp8 == 0xFF)] = 0x7E
        scales = torch.rand(experts, n // group_size, k // group_size, dtype=torch.float32) * 0.0004 + 0.0001
        return fp8.contiguous(), scales.contiguous()

    gate, gate_scales = make_fp8_weights(intermediate, hidden)
    up, up_scales = make_fp8_weights(intermediate, hidden)
    down, down_scales = make_fp8_weights(hidden, intermediate)

    pool, _, moe = amx_backend.build_moe(
        "AMXFP8_MOE",
        experts,
        hidden,
        intermediate,
        qlen,
        (gate, up, down),
        (gate_scales, up_scales, down_scales),
        bits=8,
        group_size=group_size,
    )

    # 560/40 at hidden 2048 covers several K blocks and an expert spanning two
    # 512-row accumulator chunks.
    ids, weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    result = {
        "prefill": amx_backend.forward(pool, moe, ids, weights, inputs, qlen),
        # One-token decode keeps using the matvec path regardless of the gate.
        "decode": amx_backend.forward(pool, moe, ids, weights, inputs, 1),
    }
    reference = amx_backend.swiglu_reference(
        inputs,
        ids,
        weights,
        dequantize_fp8_blockwise(gate, gate_scales, group_size),
        dequantize_fp8_blockwise(up, up_scales, group_size),
        dequantize_fp8_blockwise(down, down_scales, group_size),
    )
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("expert_counts, hidden", [((33, 31), 1024), ((63, 1), 1024), ((560, 40), 2048)])
def test_fp8_amx_prefill_matches_avx512(tmp_path, expert_counts, hidden):
    amx_backend.require_amx("AMXFP8_MOE")
    (fallback, reference), (actual, _) = amx_backend.run_both(
        __name__, tmp_path, expert_counts=expert_counts, hidden=hidden
    )

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)
    amx_backend.assert_matches_reference(actual, reference)
