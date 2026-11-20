"""INT4 W4A16 against fp32 references of its two numerics.

Experts with fewer than 8 rows (KT_W4A16_AMX_MIN_ROWS) run the exact GEMV (group sums scaled in fp32); the
others run the AMX prefill, which rounds each bf16((u - 8) * s) weight once. Scales
spread over 2^-3..2^3 per group, so a misplaced code, channel or group is far outside
the tolerances below.
"""

import pytest
import torch
from kt_kernel.utils.amx import _w4a16_regroup_scales

import amx_backend


def quantize(n, k, group_size, generator):
    """Random codes u in 0..15 packed low nibble first along K, and bf16 group scales."""
    codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.int16)
    packed = ((codes[:, 1::2] << 4) | codes[:, ::2]).to(torch.uint8).contiguous()
    exponent = torch.rand(n, k // group_size, generator=generator) * 6 - 3
    scales = (0.004 * torch.exp2(exponent)).to(torch.bfloat16).contiguous()
    return codes, packed, scales


def dequantize(codes, scales, group_size, fold):
    """(u - 8) * s in fp32, or rounded to bf16 as the AMX prefill folds it."""
    weight = (codes.float() - 8) * scales.float().repeat_interleave(group_size, dim=1)
    return weight.bfloat16().float() if fold else weight


# hidden 2048 spans two AMX K blocks of 1024. Split over four NUMA parts, intermediate
# 1024 leaves 256 columns of down's K per part and 768 leaves 192, which takes groups of
# 128 regrouped to 64 (valid for any part count).
@pytest.mark.parametrize("group_size, intermediate", [(32, 1024), (128, 1024), (128, 768)])
@pytest.mark.parametrize("expert_counts", [(33, 31), (20, 7)])
def test_int4_w4a16_matches_reference(expert_counts, group_size, intermediate):
    amx_backend.require_amx("AMXInt4W4A16_MOE")
    generator = torch.Generator().manual_seed(23)
    experts, hidden = 2, 2048
    qlen = sum(expert_counts)
    shapes = {"gate": (intermediate, hidden), "up": (intermediate, hidden), "down": (hidden, intermediate)}
    projections = {
        name: [quantize(n, k, group_size, generator) for _ in range(experts)] for name, (n, k) in shapes.items()
    }
    stack = lambda name, index: torch.stack([expert[index] for expert in projections[name]]).contiguous()
    regrouped = {
        name: _w4a16_regroup_scales([expert[2] for expert in projections[name]], group_size, intermediate // 4)
        for name in shapes
    }

    pool, _, moe = amx_backend.build_moe(
        "AMXInt4W4A16_MOE",
        experts,
        hidden,
        intermediate,
        qlen,
        tuple(stack(name, 1) for name in shapes),
        tuple(torch.stack(regrouped[name][0]).contiguous() for name in shapes),
        bits=4,
        group_size=regrouped["gate"][1],
    )
    ids, weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden, generator=generator) * 0.5).to(torch.bfloat16).contiguous()
    prefill = amx_backend.forward(pool, moe, ids, weights, inputs, qlen).float()
    decode = amx_backend.forward(pool, moe, ids, weights, inputs, 1).float()

    references = {
        fold: amx_backend.swiglu_reference(
            inputs,
            ids,
            weights,
            *[
                torch.stack([dequantize(codes, scales, group_size, fold) for codes, _, scales in projections[name]])
                for name in shapes
            ],
        )
        for fold in (False, True)
    }
    # Per expert: the AMX prefill from 8 rows, the exact GEMV below.
    expected = torch.empty_like(prefill)
    for expert, count in enumerate(expert_counts):
        rows = ids[:, 0] == expert
        expected[rows] = references[count >= 8][rows]

    def relative_error(actual, reference):
        return ((actual - reference).abs().mean() / reference.abs().mean()).item()

    # Measured 2026-11-20: 0.25-0.26% from each path's own reference, 0.46-0.51% from
    # the other numerics' one, about 130% with every scale read from the next group.
    assert relative_error(prefill, expected) < 0.0035
    assert relative_error(decode, references[False][:1]) < 0.0035
