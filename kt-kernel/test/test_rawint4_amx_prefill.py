"""Check RAWINT4 AMX prefill against the AVX512 matmat/matvec path."""

import os

import pytest
import torch

import amx_backend


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

    pool, _, moe = amx_backend.build_moe(
        backend_name,
        experts,
        hidden,
        intermediate,
        qlen,
        (gate, up, down),
        (gate_scales, up_scales, down_scales),
        bits=4,
        group_size=group_size,
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
            torch.stack([rawint4_dequantize(weight[e], scale[e], n, k, group_size) for e in range(experts)])
            for weight, scale, n, k in (
                (gate, gate_scales, intermediate, hidden),
                (up, up_scales, intermediate, hidden),
                (down, down_scales, hidden, intermediate),
            )
        ],
    )
    return result, reference


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("backend_name", ["AMXInt4_KGroup_MOE", "AMXInt4_KGroupBlocked_MOE"])
@pytest.mark.parametrize("expert_counts", [(33, 31), (63, 1)])
def test_rawint4_amx_prefill_matches_avx512(tmp_path, expert_counts, backend_name):
    amx_backend.require_amx(backend_name)
    (fallback, reference), (actual, _) = amx_backend.run_both(
        __name__, tmp_path, expert_counts=expert_counts, backend_name=backend_name
    )

    assert torch.isfinite(actual["prefill"]).all()
    torch.testing.assert_close(actual["prefill"], fallback["prefill"], atol=5e-4, rtol=5e-3)
    amx_backend.assert_matches_reference(actual, reference)
