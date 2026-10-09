"""Check BF16 AMX prefill against the AVX512 matvec path."""

import pytest
import torch
import torch.nn.functional as F

import kt_kernel


@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_bf16_amx_prefill_matches_avx512(monkeypatch, expert_counts):
    ext = kt_kernel.kt_kernel_ext
    if not hasattr(ext.moe, "AMXBF16_MOE"):
        pytest.skip("AMX BF16 extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")

    torch.manual_seed(13)
    experts, hidden, intermediate, qlen = 2, 1024, 1024, 64
    gate = (
        (torch.randn(experts, intermediate, hidden, dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )
    up = (
        (torch.randn(experts, intermediate, hidden, dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )
    down = (
        (torch.randn(experts, hidden, intermediate, dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )
    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [
        weight.data_ptr() for weight in (gate, up, down)
    ]
    config.gate_scale, config.up_scale, config.down_scale = 0, 0, 0
    config.pool = pool.backend_
    moe = ext.moe.AMXBF16_MOE(config)
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    # 33/31 exercises full and partial AMX tiles (tail rows must stay zero);
    # 63/1 exercises the single-row staging path alongside an aliased expert.
    ids = torch.tensor(
        [[0]] * expert_counts[0] + [[1]] * expert_counts[1], dtype=torch.int64
    ).contiguous()
    weights = torch.rand(qlen, 1, dtype=torch.float32).contiguous()
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    def forward(backend, target_moe=moe):
        monkeypatch.setenv("KT_BF16_PREFILL_AMX", backend)
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

    reference = torch.zeros(qlen, hidden, dtype=torch.float32)
    for expert in range(experts):
        selected = (ids[:, 0] == expert).nonzero(as_tuple=True)[0]
        gate_result = (inputs[selected].float() @ gate[expert].float().T).bfloat16().float()
        up_result = (inputs[selected].float() @ up[expert].float().T).bfloat16().float()
        activated = (F.silu(gate_result) * up_result).bfloat16().float()
        down_result = (activated @ down[expert].float().T).bfloat16().float()
        reference[selected] = weights[selected] * down_result

    error = (actual.float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02

    # One-token decode keeps using the matvec path regardless of the gate.
    batch_size[0] = 1
    decoded = forward("1")[:1]
    assert torch.isfinite(decoded).all()
    decode_error = (decoded.float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02