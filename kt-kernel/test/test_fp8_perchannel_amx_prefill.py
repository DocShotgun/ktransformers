"""Check FP8 per-channel AMX prefill against the AVX512 matvec path."""

import os

import pytest
import torch

import amx_backend


def dequantize_fp8_perchannel(fp8, scales, expert):
    """Per-channel scales are one float per output row, broadcast over K."""
    values = fp8[expert].view(torch.float8_e4m3fn).to(torch.float32)
    return values * scales[expert].view(-1, 1)


def _run(backend, expert_counts):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_FP8_PERCHANNEL_PREFILL_AMX"] = backend

    torch.manual_seed(17)
    experts, hidden, intermediate, qlen = 2, 1024, 1024, 64

    # Random FP8 codes stress every widening lookup entry (denorms, both signs);
    # kt's decode maps 0x7F/0xFF finite where torch sees NaN, so clamp those
    # codes away and let torch decode the rest exactly like the kernel's LUTs.
    # Scales are small so the reference stays O(1): the raw codes reach +-448,
    # and a unit-scale product would push the fp32 reference past 1e9 where the
    # bf16 rounding of the intermediates, not the kernel, sets the error.
    def make_fp8_weights(n, k):
        fp8 = torch.randint(0, 256, (experts, n, k), dtype=torch.int32).to(torch.uint8)
        fp8[(fp8 == 0x7F) | (fp8 == 0xFF)] = 0x7E
        scales = torch.rand(experts, n, dtype=torch.float32) * 0.0004 + 0.0001
        return fp8.contiguous(), scales.contiguous()

    gate, gate_scales = make_fp8_weights(intermediate, hidden)
    up, up_scales = make_fp8_weights(intermediate, hidden)
    down, down_scales = make_fp8_weights(hidden, intermediate)

    pool, _, moe = amx_backend.build_moe(
        "AMXFP8PerChannel_MOE",
        experts,
        hidden,
        intermediate,
        qlen,
        (gate, up, down),
        (gate_scales, up_scales, down_scales),
        bits=8,
        group_size=0,  # unused for per-channel
        per_channel=True,
    )

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
        *[
            torch.stack([dequantize_fp8_perchannel(weight, scale, e) for e in range(experts)])
            for weight, scale in ((gate, gate_scales), (up, up_scales), (down, down_scales))
        ],
    )
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_fp8_perchannel_amx_prefill_matches_avx512(tmp_path, expert_counts):
    amx_backend.require_amx("AMXFP8PerChannel_MOE")
    (fallback, reference), (actual, _) = amx_backend.run_both(__name__, tmp_path, expert_counts=expert_counts)

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)

    # assert_close alone would also pass if both backends silently took the same
    # branch, so require the two to be distinguishable. Every expert here has
    # m >= 16, so the AMX kernel must have run for at least one of them.
    assert not torch.equal(actual["prefill"], fallback["prefill"]), "AMX path did not run"
    amx_backend.assert_matches_reference(actual, reference)
