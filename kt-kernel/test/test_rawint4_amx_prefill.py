"""Check RAWINT4 AMX prefill against the AVX512 matmat/matvec path."""

import os

import pytest
import torch
import torch.nn.functional as F

import amx_backend
import kt_kernel  # noqa: F401


def rawint4_quantize(weight_bf16, quant_group_size=32):
    """Quantize [N, K] BF16 weight to RAWINT4 layout (nibble 0..15, scale amax/7)."""
    n, k = weight_bf16.shape
    assert k % 2 == 0
    assert k % quant_group_size == 0

    blocks = weight_bf16.float().view(n, k // quant_group_size, quant_group_size)
    amax = blocks.abs().amax(dim=2)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = (torch.round(blocks / scale.unsqueeze(2)) + 8).clamp(0, 15).to(torch.int16).view(n, k)
    packed = ((q[:, 1::2] << 4) | q[:, ::2]).to(torch.uint8)
    return packed.contiguous(), scale.view(n, k // quant_group_size).to(torch.bfloat16).contiguous()


def rawint4_dequantize(qweight, scales, out_features, in_features, quant_group_size=32):
    """Dequantize RAWINT4 qweight/scales back to fp32 [N, K]: (nibble - 8) * scale."""
    packed = qweight.long()
    nibbles = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=2).view(out_features, in_features)
    scale_grid = scales.float().view(out_features, in_features // quant_group_size, 1)
    scale_grid = scale_grid.expand(out_features, in_features // quant_group_size, quant_group_size)
    return (nibbles - 8).float() * scale_grid.reshape(out_features, in_features)


def _run(backend, expert_counts, backend_name):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_RAWINT4_PREFILL_AMX"] = backend
    ext = kt_kernel.kt_kernel_ext

    torch.manual_seed(17)
    experts, hidden, intermediate, qlen = 2, 1024, 1024, 64
    group_size = 32

    gate, gate_scales = rawint4_quantize(torch.randn(intermediate, hidden) * 0.02, group_size)
    up, up_scales = rawint4_quantize(torch.randn(intermediate, hidden) * 0.02, group_size)
    down, down_scales = rawint4_quantize(torch.randn(hidden, intermediate) * 0.02, group_size)
    gate = gate.unsqueeze(0).expand(experts, -1, -1).contiguous()
    up = up.unsqueeze(0).expand(experts, -1, -1).contiguous()
    down = down.unsqueeze(0).expand(experts, -1, -1).contiguous()
    gate_scales = gate_scales.unsqueeze(0).expand(experts, -1, -1).contiguous()
    up_scales = up_scales.unsqueeze(0).expand(experts, -1, -1).contiguous()
    down_scales = down_scales.unsqueeze(0).expand(experts, -1, -1).contiguous()

    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.quant_config.bits = 4
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [
        weight.data_ptr() for weight in (gate, up, down)
    ]
    config.gate_scale, config.up_scale, config.down_scale = [
        scale.data_ptr() for scale in (gate_scales, up_scales, down_scales)
    ]
    config.pool = pool.backend_
    moe = getattr(ext.moe, backend_name)(config)
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    # 33/31 exercises full and partial AMX tiles (tail rows must stay zero);
    # 63/1 exercises the single-row staging fallback alongside an aliased expert.
    ids = torch.tensor(
        [[0]] * expert_counts[0] + [[1]] * expert_counts[1], dtype=torch.int64
    ).contiguous()
    weights = torch.rand(qlen, 1, dtype=torch.float32).contiguous()
    inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    def forward():
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

    result = {"prefill": forward()}

    gate_deq = rawint4_dequantize(gate[0], gate_scales[0], intermediate, hidden, group_size)
    up_deq = rawint4_dequantize(up[0], up_scales[0], intermediate, hidden, group_size)
    down_deq = rawint4_dequantize(down[0], down_scales[0], hidden, intermediate, group_size)

    selected = torch.arange(qlen)
    gate_result = (inputs[selected].float() @ gate_deq.T).bfloat16().float()
    up_result = (inputs[selected].float() @ up_deq.T).bfloat16().float()
    activated = (F.silu(gate_result) * up_result).bfloat16().float()
    down_result = (activated @ down_deq.T).bfloat16().float()
    reference = weights[selected] * down_result

    # One-token decode keeps using the matvec path regardless of the gate.
    batch_size[0] = 1
    result["decode"] = forward()[:1]
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("backend_name", ["AMXInt4_KGroup_MOE", "AMXInt4_KGroupBlocked_MOE"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_rawint4_amx_prefill_matches_avx512(tmp_path, expert_counts, backend_name):
    ext = kt_kernel.kt_kernel_ext
    if getattr(ext.moe, backend_name, None) is None:
        pytest.skip(f"{backend_name} extension is unavailable")
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
            backend_name=backend_name,
        )
        for backend in ("0", "1")
    }
    fallback, reference = results["0"]
    actual = results["1"][0]

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)

    error = (actual["prefill"].float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02

    assert torch.isfinite(actual["decode"]).all()
    decode_error = (actual["decode"].float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02
