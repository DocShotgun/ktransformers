"""Check MXFP4 AMX, vector, and weight-layout numerical agreement."""

import os

import pytest
import torch

import amx_backend


def unpack_mxfp4(packed, scales):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    unscaled = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return unscaled * scales.float().repeat_interleave(32, -1)


def _run(backend, expert_counts, weight_layout):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_MXFP4_KMAJOR_WEIGHTS"] = weight_layout
    os.environ["KT_MXFP4_PREFILL_AMX"] = backend

    torch.manual_seed(13)
    experts, hidden, intermediate, qlen = 2, 1024, 1024, 64
    packed = [
        torch.randint(0, 256, (experts, intermediate, hidden // 2), dtype=torch.uint8).contiguous()
        for _ in range(3)
    ]
    scales = [
        (2.0 ** torch.randint(-6, -3, (experts, intermediate, hidden // 32)))
        .to(torch.bfloat16)
        .contiguous()
        for _ in range(3)
    ]
    pool, config, moe = amx_backend.build_moe(
        "AMXFP4_KGroup_MOE", experts, hidden, intermediate, qlen, packed, scales, bits=4, group_size=32
    )

    ids, weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    result = {
        "prefill": amx_backend.forward(pool, moe, ids, weights, inputs, qlen),
        # One-token decode uses GEMV directly on the resident K-major layout.
        "decode": amx_backend.forward(pool, moe, ids, weights, inputs, 1),
    }

    if weight_layout == "1":
        # Both layouts must agree on the same random weights and routes.
        os.environ["KT_MXFP4_KMAJOR_WEIGHTS"] = "0"
        row_moe = type(moe)(config)
        amx_backend.load_weights(pool, row_moe, experts)
        result["row_prefill"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, qlen)
        result["row_decode"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, 1)
        # Short prompts route only a few tokens per expert through GEMV.
        # Their activations are in ordinary K order, unlike single-token decode.
        result["small_prefill"] = amx_backend.forward(pool, moe, ids, weights, inputs, 8)
        result["row_small_prefill"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, 8)

    reference = amx_backend.swiglu_reference(
        inputs, ids, weights, *[unpack_mxfp4(weight, scale) for weight, scale in zip(packed, scales)]
    )
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("weight_layout", ["0", "1"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_mxfp4_amx_prefill_matches_avx512(tmp_path, expert_counts, weight_layout):
    amx_backend.require_amx("AMXFP4_KGroup_MOE")
    # The two kernels come out bit-identical here: MXFP4's power-of-two scales
    # and small group size keep every partial sum exact in fp32, so this only
    # pins the kernels against each other, not against rounding order.
    (fallback, reference), (amx, _) = amx_backend.run_both(
        __name__, tmp_path, expert_counts=expert_counts, weight_layout=weight_layout
    )

    assert torch.isfinite(amx["prefill"]).all()
    torch.testing.assert_close(amx["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)
    amx_backend.assert_matches_reference(amx, reference)

    if weight_layout == "1":
        for key in ("prefill", "decode", "small_prefill"):
            torch.testing.assert_close(amx[key], amx[f"row_{key}"], atol=5e-4, rtol=5e-3)
