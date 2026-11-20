"""INT4 W4A16 against fp32 references of its two numerics.

Experts with fewer than 8 rows (KT_W4A16_AMX_MIN_ROWS) run the exact GEMV (group sums
scaled in fp32); the others run the AMX prefill, which rounds each bf16((u - z) * s)
weight once. Scales spread over 2^-3..2^3 per group and zero codes are random, so a
misplaced code, zero, channel or group is far outside the tolerance below.
"""

import glob
import json

import pytest
import torch
from kt_kernel.utils import w4a16
from safetensors.torch import save_file

import amx_backend

# Measured 2026-11-20: 0.25-0.29% from each path's own reference, 0.46-0.53% from the
# other numerics' one; about 130% with every scale or zero read from the next group,
# about 94% with the zeros ignored.
TOLERANCE = 0.0035


def random_projection(n, k, group_size, asymmetric, generator):
    """Codes u in 0..15, zero codes z (8 when symmetric) and bf16 group scales."""
    groups = k // group_size
    codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.int16)
    if asymmetric:
        zeros = torch.randint(0, 16, (n, groups), generator=generator, dtype=torch.int16)
    else:
        zeros = torch.full((n, groups), 8, dtype=torch.int16)
    exponent = torch.rand(n, groups, generator=generator) * 6 - 3
    scales = (0.004 * torch.exp2(exponent)).to(torch.bfloat16).contiguous()
    return codes, zeros, scales


def pack(codes):
    """Codes [n, k] to bytes [n, k / 2], low nibble first along K."""
    return ((codes[:, 1::2] << 4) | codes[:, ::2]).to(torch.uint8).contiguous()


def dequantize(codes, zeros, scales, group_size, fold):
    """(u - z) * s in fp32, or rounded to bf16 as the AMX prefill folds it."""
    expand = lambda t: t.float().repeat_interleave(group_size, dim=1)
    weight = (codes.float() - expand(zeros)) * expand(scales)
    return weight.bfloat16().float() if fold else weight


def check_outputs(prefill, decode, inputs, ids, route_weights, expert_counts, dequantized):
    """Per expert, the AMX prefill (from 8 rows) against the folded reference and the GEMV
    against the exact one; one-token decode against the exact one."""
    references = {
        fold: amx_backend.swiglu_reference(inputs, ids, route_weights, *dequantized(fold)) for fold in (False, True)
    }
    expected = torch.empty_like(prefill)
    for expert, count in enumerate(expert_counts):
        rows = ids[:, 0] == expert
        expected[rows] = references[count >= 8][rows]

    def relative_error(actual, reference):
        return ((actual - reference).abs().mean() / reference.abs().mean()).item()

    assert relative_error(prefill, expected) < TOLERANCE
    assert relative_error(decode, references[False][:1]) < TOLERANCE


# hidden 2048 spans two AMX K blocks of 1024. Split over four NUMA parts, intermediate
# 1024 leaves 256 columns of down's K per part and 768 leaves 192, which takes groups of
# 128 regrouped to 64 (valid for any part count).
@pytest.mark.parametrize(
    "group_size, intermediate, asymmetric", [(32, 1024, False), (128, 1024, True), (128, 768, True)]
)
@pytest.mark.parametrize("expert_counts", [(33, 31), (20, 7)])
def test_int4_w4a16_matches_reference(expert_counts, group_size, intermediate, asymmetric):
    amx_backend.require_amx("AMXInt4W4A16_MOE")
    generator = torch.Generator().manual_seed(23)
    experts, hidden = 2, 2048
    qlen = sum(expert_counts)
    shapes = {"gate": (intermediate, hidden), "up": (intermediate, hidden), "down": (hidden, intermediate)}
    projections = {
        name: [random_projection(n, k, group_size, asymmetric, generator) for _ in range(experts)]
        for name, (n, k) in shapes.items()
    }
    target = w4a16.part_group_size([group_size], intermediate // 4)

    def stacked(name, index, dtype):
        regrouped = w4a16.regroup([expert[index] for expert in projections[name]], group_size, target)
        return torch.stack(regrouped).to(dtype).contiguous()

    pool, _, moe = amx_backend.build_moe(
        "AMXInt4W4A16_MOE",
        experts,
        hidden,
        intermediate,
        qlen,
        tuple(torch.stack([pack(expert[0]) for expert in projections[name]]).contiguous() for name in shapes),
        tuple(stacked(name, 2, torch.bfloat16) for name in shapes),
        bits=4,
        group_size=target,
        zeros=tuple(stacked(name, 1, torch.uint8) for name in shapes) if asymmetric else None,
    )
    ids, route_weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden, generator=generator) * 0.5).to(torch.bfloat16).contiguous()
    prefill = amx_backend.forward(pool, moe, ids, route_weights, inputs, qlen).float()
    decode = amx_backend.forward(pool, moe, ids, route_weights, inputs, 1).float()
    check_outputs(
        prefill,
        decode,
        inputs,
        ids,
        route_weights,
        expert_counts,
        lambda fold: [
            torch.stack([dequantize(*expert, group_size, fold) for expert in projections[name]]) for name in shapes
        ],
    )


def write_checkpoint(path, experts, shapes, strategy, asymmetric, generator):
    """A one-layer checkpoint packed by compressed-tensors' own compressor; returns each
    projection's fp32 (u - z) * s weights."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    args = QuantizationArgs(
        num_bits=4,
        type="int",
        strategy=strategy,
        group_size=128 if strategy == "group" else None,
        symmetric=not asymmetric,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    tensors, weights = {}, {name: [] for name in shapes}
    for expert in range(experts):
        for name, (n, k) in shapes.items():
            group_size = 128 if strategy == "group" else k
            codes, zeros, scales = random_projection(n, k, group_size, asymmetric, generator)
            weight = dequantize(codes, zeros, scales, group_size, fold=False)
            # The library's int8 values are q = u - 8 and zp = z - 8.
            state = {
                "weight": weight.bfloat16(),
                "weight_scale": scales,
                "weight_zero_point": (zeros - 8).to(torch.int8),
            }
            compressed = PackedQuantizationCompressor.compress(state, scheme)
            restored = PackedQuantizationCompressor.decompress(compressed, scheme)["weight"]
            assert torch.equal(restored.float(), weight.bfloat16().float())
            prefix = f"model.layers.0.mlp.experts.{expert}.{name}_proj"
            tensors.update({f"{prefix}.{key}": value.contiguous() for key, value in compressed.items()})
            weights[name].append(weight)
    save_file(tensors, str(path / "model.safetensors"))
    group = {"targets": ["Linear"], "weights": json.loads(args.model_dump_json())}
    quant = {"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups": {"group_0": group}}
    (path / "config.json").write_text(json.dumps({"quantization_config": quant}))
    return weights


@pytest.mark.parametrize("strategy, asymmetric", [("group", True), ("channel", False)])
def test_int4_w4a16_reads_compressed_tensors_checkpoints(tmp_path, strategy, asymmetric):
    pytest.importorskip("compressed_tensors")
    amx_backend.require_amx("AMXInt4W4A16_MOE")
    from kt_kernel import KTMoEWrapper

    generator = torch.Generator().manual_seed(29)
    # Expert 2 stays on the GPU: the loader skips it and no route reaches it. intermediate
    # 768 regroups both strategies' scales over four NUMA parts.
    experts, hidden, intermediate, expert_counts = 3, 2048, 768, (33, 7)
    shapes = {"gate": (intermediate, hidden), "up": (intermediate, hidden), "down": (hidden, intermediate)}
    weights = write_checkpoint(tmp_path, experts, shapes, strategy, asymmetric, generator)
    nodes = max(1, len(glob.glob("/sys/devices/system/node/node[0-9]*")))
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=experts,
        num_experts_per_tok=1,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        gpu_experts_mask=torch.tensor([False, False, True]),
        cpuinfer_threads=4 * nodes,
        threadpool_count=nodes,
        weight_path=str(tmp_path),
        chunked_prefill_size=64,
        method="W4A16",
        max_deferred_experts_per_token=0,
    )
    wrapper.load_weights(torch.arange(experts, dtype=torch.int64))

    qlen = sum(expert_counts)
    ids, route_weights = amx_backend.two_expert_routes(expert_counts)
    inputs = (torch.randn(qlen, hidden, generator=generator) * 0.5).to(torch.bfloat16).contiguous()
    prefill = amx_backend.forward(wrapper.cpu_infer, wrapper.moe, ids, route_weights, inputs, qlen).float()
    decode = amx_backend.forward(wrapper.cpu_infer, wrapper.moe, ids, route_weights, inputs, 1).float()
    check_outputs(
        prefill,
        decode,
        inputs,
        ids,
        route_weights,
        expert_counts,
        lambda fold: [
            torch.stack([weight.bfloat16().float() if fold else weight for weight in weights[name][:2]])
            for name in shapes
        ],
    )
