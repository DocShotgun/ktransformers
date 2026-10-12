"""Check FP8 per-channel AMX prefill against the AVX512 matvec path."""

import os

import pytest
import torch
import torch.nn.functional as F

import amx_backend
import kt_kernel  # noqa: F401


def dequantize_fp8_perchannel(fp8, scales, expert):
    """Per-channel scales are one float per output row, broadcast over K."""
    values = fp8[expert].view(torch.float8_e4m3fn).to(torch.float32)
    return values * scales[expert].view(-1, 1)


def _run(backend, expert_counts):
    """Run one prefill backend and return its outputs plus the fp32 reference."""
    os.environ["KT_FP8_PERCHANNEL_PREFILL_AMX"] = backend
    ext = kt_kernel.kt_kernel_ext

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

    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.quant_config.bits = 8
    config.quant_config.group_size = 0  # unused for per-channel
    config.quant_config.zero_point = False
    config.quant_config.per_channel = True
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [
        weight.data_ptr() for weight in (gate, up, down)
    ]
    config.gate_scale, config.up_scale, config.down_scale = [
        scale.data_ptr() for scale in (gate_scales, up_scales, down_scales)
    ]
    config.pool = pool.backend_
    moe = ext.moe.AMXFP8PerChannel_MOE(config)
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

    gate_deq = [dequantize_fp8_perchannel(gate, gate_scales, e) for e in range(experts)]
    up_deq = [dequantize_fp8_perchannel(up, up_scales, e) for e in range(experts)]
    down_deq = [dequantize_fp8_perchannel(down, down_scales, e) for e in range(experts)]

    reference = torch.zeros(qlen, hidden, dtype=torch.float32)
    for expert in range(experts):
        selected = (ids[:, 0] == expert).nonzero(as_tuple=True)[0]
        gate_result = (inputs[selected].float() @ gate_deq[expert].T).bfloat16().float()
        up_result = (inputs[selected].float() @ up_deq[expert].T).bfloat16().float()
        activated = (F.silu(gate_result) * up_result).bfloat16().float()
        down_result = (activated @ down_deq[expert].T).bfloat16().float()
        reference[selected] = weights[selected] * down_result

    # One-token decode keeps using the matvec path regardless of the gate.
    batch_size[0] = 1
    result["decode"] = forward()[:1]
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_fp8_perchannel_amx_prefill_matches_avx512(tmp_path, expert_counts):
    ext = kt_kernel.kt_kernel_ext
    if not hasattr(ext.moe, "AMXFP8PerChannel_MOE"):
        pytest.skip("AMX FP8 per-channel extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")

    # Kernel selection is cached per process, so each backend gets its own.
    results = {
        backend: amx_backend.run_backend(
            __name__, backend, tmp_path, expert_counts=expert_counts
        )
        for backend in ("0", "1")
    }
    fallback, reference = results["0"]
    actual = results["1"][0]

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)

    # assert_close alone would also pass if both backends silently took the same
    # branch, so require the two to be distinguishable. Every expert here has
    # m >= 16, so the AMX kernel must have run for at least one of them.
    assert not torch.equal(actual["prefill"], fallback["prefill"]), "AMX path did not run"

    error = (actual["prefill"].float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02

    assert torch.isfinite(actual["decode"]).all()
    decode_error = (actual["decode"].float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02
