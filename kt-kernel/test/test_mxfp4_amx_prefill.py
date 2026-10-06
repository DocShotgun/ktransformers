"""Check MXFP4 AMX, vector, and weight-layout numerical agreement."""

import pytest
import torch
import torch.nn.functional as F

import kt_kernel


def unpack_mxfp4(packed, scales):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    unscaled = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return unscaled * scales.float().repeat_interleave(32, -1)


@pytest.mark.parametrize("weight_layout", ["0", "1"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_mxfp4_amx_prefill_matches_avx512(monkeypatch, expert_counts, weight_layout):
    monkeypatch.setenv("KT_MXFP4_KMAJOR_WEIGHTS", weight_layout)
    ext = kt_kernel.kt_kernel_ext
    if not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX MXFP4 extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")

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
    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [weight.data_ptr() for weight in packed]
    config.gate_scale, config.up_scale, config.down_scale = [scale.data_ptr() for scale in scales]
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
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

    def forward(backend, target_moe=moe):
        monkeypatch.setenv("KT_MXFP4_PREFILL_AMX", backend)
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

    expected = forward("0")
    actual = forward("1")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-3)

    gate, up, down = [
        unpack_mxfp4(weight, scale) for weight, scale in zip(packed, scales)
    ]
    reference = torch.zeros(qlen, hidden, dtype=torch.float32)
    for expert in range(experts):
        selected = (ids[:, 0] == expert).nonzero(as_tuple=True)[0]
        gate_result = (inputs[selected].float() @ gate[expert].T).bfloat16().float()
        up_result = (inputs[selected].float() @ up[expert].T).bfloat16().float()
        activated = (F.silu(gate_result) * up_result).bfloat16().float()
        down_result = (activated @ down[expert].T).bfloat16().float()
        reference[selected] = weights[selected] * down_result

    error = (actual.float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02

    # One-token decode uses GEMV directly on the resident K-major layout.
    batch_size[0] = 1
    decoded = forward("1")[:1]
    assert torch.isfinite(decoded).all()
    decode_error = (decoded.float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02

    if weight_layout == "1":
        # Both layouts must agree on the same random weights and routes.
        monkeypatch.setenv("KT_MXFP4_KMAJOR_WEIGHTS", "0")
        row_moe = ext.moe.AMXFP4_KGroup_MOE(config)
        pool.submit(row_moe.load_weights_task(mapping.data_ptr()))
        pool.sync()
        batch_size[0] = qlen
        row_prefill = forward("1", row_moe)
        torch.testing.assert_close(actual, row_prefill, atol=5e-4, rtol=5e-3)
        batch_size[0] = 1
        row_decode = forward("1", row_moe)[:1]
        torch.testing.assert_close(decoded, row_decode, atol=5e-4, rtol=5e-3)
        # Short prompts route only a few tokens per expert through GEMV.
        # Their activations are in ordinary K order, unlike single-token decode.
        batch_size[0] = 8
        small_prefill = forward("1")[:8]
        row_small_prefill = forward("1", row_moe)[:8]
        torch.testing.assert_close(small_prefill, row_small_prefill, atol=5e-4, rtol=5e-3)
