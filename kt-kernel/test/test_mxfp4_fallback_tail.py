"""Sweep the AVX-512 fallback across M/N tile boundaries.

`fp4_mat_mat_kgroup` tiles M and N by 4 with scalar tail loops.  The AMX
comparison tests only exercise a single prompt length, so sweep the shapes that
land in the tails here and check the fallback against the AMX kernel.
"""

import pytest
import torch

import kt_kernel


@pytest.mark.parametrize("weight_layout", ["0", "1"])
@pytest.mark.parametrize("qlen", [17, 18, 19, 20, 33, 63, 65, 67])
def test_mxfp4_fallback_tile_tails(monkeypatch, qlen, weight_layout):
    monkeypatch.setenv("KT_MXFP4_KMAJOR_WEIGHTS", weight_layout)
    ext = kt_kernel.kt_kernel_ext
    if not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX MXFP4 extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")

    torch.manual_seed(qlen * 7 + int(weight_layout))
    experts, hidden, intermediate = 2, 256, 512
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
    config.gate_proj, config.up_proj, config.down_proj = [w.data_ptr() for w in packed]
    config.gate_scale, config.up_scale, config.down_scale = [s.data_ptr() for s in scales]
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.pool = pool.backend_
    moe = ext.moe.AMXFP4_KGroup_MOE(config)
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    # Split the batch across both experts and leave the last expert partial so
    # the M tail is non-empty for every qlen.
    split = max(1, qlen // 2)
    ids = torch.tensor([[0]] * split + [[1]] * (qlen - split), dtype=torch.int64).contiguous()
    weights = torch.rand(qlen, 1, dtype=torch.float32).contiguous()
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    def forward(backend):
        monkeypatch.setenv("KT_MXFP4_PREFILL_AMX", backend)
        output = torch.empty(qlen, hidden, dtype=torch.bfloat16)
        pool.submit(
            moe.forward_task(
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

    expected = forward("1")
    actual = forward("0")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-3)
