"""Check NVFP4 AMX, vector, and weight-layout numerical agreement.

NVFP4 shares MXFP4's E2M1 nibble packing but carries one E4M3x-global scale
per 16 K values (folded to bf16 by the loader) instead of one ue8m0 scale per
32. All AMX prefill / AVX512 decode / weight-layout combinations must agree
with a float32 dequantization reference.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import amx_backend
import kt_kernel  # noqa: F401


def unpack_nvfp4(packed, scales):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    unscaled = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return unscaled * scales.float().repeat_interleave(16, -1)


def make_nvfp4_rows(experts, rows, cols):
    packed = torch.randint(
        0, 256, (experts, rows, cols // 2), dtype=torch.uint8
    ).contiguous()
    # bf16-folded scales must survive the kernel's exact-roundtrip validation,
    # and should not degrade into powers of two (which would let an e8 path
    # mask NVFP4-specific bugs).
    scales = (
        (torch.rand(experts, rows, cols // 16) * 1.5 + 0.25)
        * (2.0 ** torch.randint(-6, -3, (experts, rows, cols // 16)))
    ).to(torch.bfloat16).contiguous()
    return packed, scales


def _run(backend, expert_counts, weight_layout, hidden, intermediate):
    """Run one prefill backend and return every tensor the test compares."""
    os.environ["KT_NVFP4_KMAJOR_WEIGHTS"] = weight_layout
    os.environ["KT_NVFP4_PREFILL_AMX"] = backend
    ext = kt_kernel.kt_kernel_ext

    torch.manual_seed(17)
    experts, qlen = 2, 64
    gate_packed, gate_scale = make_nvfp4_rows(experts, intermediate, hidden)
    up_packed, up_scale = make_nvfp4_rows(experts, intermediate, hidden)
    down_packed, down_scale = make_nvfp4_rows(experts, hidden, intermediate)
    packed = [gate_packed, up_packed, down_packed]
    scales = [gate_scale, up_scale, down_scale]

    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [
        weight.data_ptr() for weight in packed
    ]
    config.gate_scale, config.up_scale, config.down_scale = [
        scale.data_ptr() for scale in scales
    ]
    config.quant_config.bits = 4
    config.quant_config.group_size = 16
    config.quant_config.zero_point = False
    config.pool = pool.backend_
    moe = ext.moe.AMXFP4_KGroup_MOE(config)
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    # 33/31 exercises full and partial AMX tiles; 63/1 also exercises the
    # single-row staging fallback alongside an aliased prefill expert.
    ids = torch.tensor(
        [[0]] * expert_counts[0] + [[1]] * expert_counts[1], dtype=torch.int64
    ).contiguous()
    weights = torch.rand(qlen, 1, dtype=torch.float32).contiguous()
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    def forward(target_moe):
        output = torch.empty(qlen, hidden, dtype=torch.bfloat16)
        pool.submit(
            target_moe.forward_task(
                batch_size.data_ptr(),
                1,
                ids.data_ptr(),
                weights.data_ptr(),
                inputs.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        pool.sync()
        return output

    result = {"prefill": forward(moe)}

    # One-token decode uses GEMV directly on the resident K-major layout.
    batch_size[0] = 1
    result["decode"] = forward(moe)[:1]

    if weight_layout == "1":
        # Both layouts must agree on the same random weights and routes.
        # Each layout folds scales identically -- bf16(fp4*scale) -- through
        # independent packers, so agreement is near bit-close; keep a small
        # slack budget for independent fold-path assembly while each layout
        # separately tracks the float32 reference in the test body.
        os.environ["KT_NVFP4_KMAJOR_WEIGHTS"] = "0"
        row_moe = ext.moe.AMXFP4_KGroup_MOE(config)
        pool.submit(row_moe.load_weights_task(mapping.data_ptr()))
        pool.sync()
        batch_size[0] = qlen
        result["row_prefill"] = forward(row_moe)
        batch_size[0] = 1
        result["row_decode"] = forward(row_moe)[:1]
        # Short prompts route only a few tokens per expert through GEMV.
        # Their activations are in ordinary K order, unlike single-token decode.
        batch_size[0] = 8
        result["small_prefill"] = forward(moe)[:8]
        result["row_small_prefill"] = forward(row_moe)[:8]

    return result, packed, scales, inputs, ids, weights


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("weight_layout", ["0", "1"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
@pytest.mark.parametrize("hidden,intermediate", [(1024, 1024), (1024, 1152)])
def test_nvfp4_amx_prefill_matches_avx512(
    tmp_path, expert_counts, weight_layout, hidden, intermediate
):
    ext = kt_kernel.kt_kernel_ext
    if not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX FP4 extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")

    # Kernel selection is cached per process, so each backend gets its own.
    results = {
        backend: amx_backend.run_backend(
            __name__,
            backend,
            tmp_path,
            expert_counts=expert_counts,
            weight_layout=weight_layout,
            hidden=hidden,
            intermediate=intermediate,
        )
        for backend in ("0", "1")
    }
    fallback, packed, scales, inputs, ids, weights = results["0"]
    amx = results["1"][0]

    actual = amx["prefill"]
    assert torch.isfinite(actual).all()
    # NVFP4 AMX folds the per-16 scales into bf16(fp4*scale) while the AVX512
    # path applies scales in fp32 after each dot: one bf16 product rounding per
    # K contribution accumulates to ~8e-3 max abs at qlen 64 / k <= 1152 (and
    # grows with k). Budget that fold explicitly here; the float32-reference
    # check below remains the accuracy standard (~0.5% mean error measured
    # against a 2% budget).
    torch.testing.assert_close(actual, fallback["prefill"], atol=2e-2, rtol=5e-3)

    qlen = inputs.shape[0]
    gate, up, down = [
        unpack_nvfp4(weight, scale) for weight, scale in zip(packed, scales)
    ]
    reference = torch.zeros(qlen, hidden, dtype=torch.float32)
    for expert in range(2):
        selected = (ids[:, 0] == expert).nonzero(as_tuple=True)[0]
        gate_result = (inputs[selected].float() @ gate[expert].T).bfloat16().float()
        up_result = (inputs[selected].float() @ up[expert].T).bfloat16().float()
        activated = (F.silu(gate_result) * up_result).bfloat16().float()
        down_result = (activated @ down[expert].T).bfloat16().float()
        reference[selected] = weights[selected] * down_result

    error = (actual.float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02

    decoded = amx["decode"]
    assert torch.isfinite(decoded).all()
    decode_error = (decoded.float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02

    if weight_layout == "1":
        for key in ("prefill", "decode", "small_prefill"):
            torch.testing.assert_close(
                amx[key], amx[f"row_{key}"], atol=2e-3, rtol=5e-3
            )
