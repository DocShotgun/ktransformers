"""Check NVFP4 AMX, vector, and weight-layout numerical agreement.

NVFP4 shares MXFP4's E2M1 nibble packing but carries one E4M3 block scale per
16 K values times a per-tensor fp32 global, instead of one ue8m0 scale per 32.
The block scales go in as exact bf16 e4m3 values and the globals separately
(MOEConfig.*_global_scales), as the AMX wrapper loads real checkpoints. All AMX
prefill / AVX512 decode / weight-layout combinations must agree with a float32
dequantization reference, and AMX with AVX512 to bit-close parity.
"""

import os

import pytest
import torch

import amx_backend


def unpack_nvfp4(packed, scales):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    unscaled = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return unscaled * scales.float().repeat_interleave(16, -1)


def make_nvfp4_rows(experts, rows, cols, pow2_scales=False):
    packed = torch.randint(
        0, 256, (experts, rows, cols // 2), dtype=torch.uint8
    ).contiguous()
    # Real e4m3 block scales (exact in bf16) and non-power-of-two globals, like
    # GLM-5.3's. All-power-of-two block scales must not select the MXFP4 (E8)
    # layout, whose kernels assume one scale per 32 K; 8..128 keeps the outputs
    # in the range where assert_amx_matches_avx512's atol is meaningful.
    raw = (
        2.0 ** torch.randint(3, 8, (experts, rows, cols // 16)).float()
        if pow2_scales
        else torch.rand(experts, rows, cols // 16) * 216 + 8
    )
    block = raw.to(torch.float8_e4m3fn).float().to(torch.bfloat16).contiguous()
    globals_ = ((torch.rand(experts) * 0.5 + 0.75) * 2.0**-11).tolist()
    return packed, block, globals_


def assert_amx_matches_avx512(actual, expected):
    """AMX vs AVX512 at the shared FP4 parity tolerance (atol 5e-4, rtol 5e-3).

    bf16(fp4 * e4m3) is exact, so both paths compute the same products; they sum
    them in different orders, so a rare bf16 rounding flip in an intermediate
    (<= 0.03% of outputs measured, one bf16 step) is the only allowed deviation.
    Folding the global into the block scales breaks ~35% of outputs and 3e-2.
    """
    diff = (actual.float() - expected.float()).abs()
    outside = diff > 5e-4 + 5e-3 * expected.float().abs()
    assert outside.float().mean().item() <= 1e-3, f"{int(outside.sum())} outputs outside tolerance"
    assert diff.max().item() <= 1.6e-2, f"max abs diff {diff.max().item():.3e}"


def _run(backend, expert_counts, weight_layout, hidden, intermediate, pow2_scales=False):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_NVFP4_KMAJOR_WEIGHTS"] = weight_layout
    os.environ["KT_NVFP4_PREFILL_AMX"] = backend

    torch.manual_seed(17)
    experts, qlen = 2, 64
    gate_packed, gate_scale, gate_global = make_nvfp4_rows(experts, intermediate, hidden, pow2_scales)
    up_packed, up_scale, up_global = make_nvfp4_rows(experts, intermediate, hidden, pow2_scales)
    down_packed, down_scale, down_global = make_nvfp4_rows(experts, hidden, intermediate, pow2_scales)
    packed = [gate_packed, up_packed, down_packed]
    scales = [gate_scale, up_scale, down_scale]
    global_scales = [gate_global, up_global, down_global]

    gate_global_scales, up_global_scales, down_global_scales = global_scales
    pool, config, moe = amx_backend.build_moe(
        "AMXFP4_KGroup_MOE",
        experts,
        hidden,
        intermediate,
        qlen,
        packed,
        scales,
        bits=4,
        group_size=16,
        gate_global_scales=gate_global_scales,
        up_global_scales=up_global_scales,
        down_global_scales=down_global_scales,
    )

    ids, weights = amx_backend.two_expert_routes(expert_counts)
    # Keep the GEMM outputs at the 1024-wide magnitude the absolute tolerances
    # in assert_amx_matches_avx512 were calibrated on.
    inputs = (torch.randn(qlen, hidden) * 0.1 * (1024 / hidden) ** 0.5).to(torch.bfloat16).contiguous()
    result = {
        "prefill": amx_backend.forward(pool, moe, ids, weights, inputs, qlen),
        # One-token decode uses GEMV directly on the resident K-major layout.
        "decode": amx_backend.forward(pool, moe, ids, weights, inputs, 1),
    }

    if weight_layout == "1":
        # Both layouts must agree on the same random weights and routes; each
        # also tracks the float32 reference separately in the test body.
        os.environ["KT_NVFP4_KMAJOR_WEIGHTS"] = "0"
        row_moe = type(moe)(config)
        amx_backend.load_weights(pool, row_moe, experts)
        result["row_prefill"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, qlen)
        result["row_decode"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, 1)
        # Short prompts route only a few tokens per expert through GEMV.
        # Their activations are in ordinary K order, unlike single-token decode.
        result["small_prefill"] = amx_backend.forward(pool, moe, ids, weights, inputs, 8)
        result["row_small_prefill"] = amx_backend.forward(pool, row_moe, ids, weights, inputs, 8)

    reference = amx_backend.swiglu_reference(
        inputs,
        ids,
        weights,
        *[
            unpack_nvfp4(weight, scale) * torch.tensor(glob).view(-1, 1, 1)
            for weight, scale, glob in zip(packed, scales, global_scales)
        ],
    )
    return result, reference


def child_main():
    amx_backend.child_main(_run)


def _check(fallback, reference, amx):
    assert torch.isfinite(amx["prefill"]).all()
    assert_amx_matches_avx512(amx["prefill"], fallback["prefill"])
    amx_backend.assert_matches_reference(amx, reference)


@pytest.mark.parametrize("weight_layout", ["0", "1"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
# hidden 8192 > K_BLOCK (7168): one row's scales (k / 16 = 512 groups) span
# more than a K block.
@pytest.mark.parametrize("hidden,intermediate", [(1024, 1024), (1024, 1152), (8192, 256)])
def test_nvfp4_amx_prefill_matches_avx512(
    tmp_path, expert_counts, weight_layout, hidden, intermediate
):
    amx_backend.require_amx("AMXFP4_KGroup_MOE")
    (fallback, reference), (amx, _) = amx_backend.run_both(
        __name__,
        tmp_path,
        expert_counts=expert_counts,
        weight_layout=weight_layout,
        hidden=hidden,
        intermediate=intermediate,
    )
    _check(fallback, reference, amx)

    if weight_layout == "1":
        for key in ("prefill", "decode", "small_prefill"):
            torch.testing.assert_close(
                amx[key], amx[f"row_{key}"], atol=2e-3, rtol=5e-3
            )


@pytest.mark.parametrize("weight_layout", ["0", "1"])
def test_nvfp4_power_of_two_block_scales(tmp_path, weight_layout):
    """All-power-of-two NVFP4 block scales must give the same results as e4m3 ones."""
    amx_backend.require_amx("AMXFP4_KGroup_MOE")
    (fallback, reference), (amx, _) = amx_backend.run_both(
        __name__,
        tmp_path,
        expert_counts=(33, 31),
        weight_layout=weight_layout,
        hidden=1024,
        intermediate=1024,
        pow2_scales=True,
    )
    _check(fallback, reference, amx)
