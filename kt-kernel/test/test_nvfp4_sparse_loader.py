"""NVFP4 loader: sparse GPU-expert skip, BF16 scale folding."""

import pytest
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    pytest.skip("safetensors is unavailable", allow_module_level=True)

from kt_kernel.utils.loader import NVFP4SafeTensorLoader

# Experts prefix under base_key "model.layers.7", (gate, up, down), block scale suffix.
PREFIX, PROJ_NAMES, BLOCK_SUFFIX = "model.layers.7.mlp.experts", ("gate_proj", "up_proj", "down_proj"), "weight_scale"
FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def make_checkpoint(
    tmp_path,
    global_suffix="weight_scale_2",
    expert_count=3,
    hidden=64,
    intermediate=128,
):
    down_name = PROJ_NAMES[2]
    tensors = {}
    for i in range(expert_count):
        for proj in PROJ_NAMES:
            rows, cols = (intermediate, hidden) if proj != down_name else (hidden, intermediate)
            pfx = f"{PREFIX}.{i}.{proj}"
            tensors[f"{pfx}.weight"] = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
            tensors[f"{pfx}.{BLOCK_SUFFIX}"] = torch.randn(rows, cols // 16).clamp(-4, 4).to(torch.float8_e4m3fn)
            if global_suffix is not None:
                tensors[f"{pfx}.{global_suffix}"] = torch.tensor(0.0123, dtype=torch.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def unpack_fp4(packed, scales):
    values = torch.tensor(FP4_VALUES, dtype=torch.float64)
    unscaled = torch.stack((values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1).flatten(-2)
    return unscaled * scales


def test_nvfp4_sparse_gpu_expert_skip(tmp_path):
    make_checkpoint(tmp_path)
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    mask = torch.tensor([False, True, False])
    weights = loader.load_experts("model.layers.7", gpu_experts_mask=mask)

    for key in ("gate", "up", "down", "gate_scale", "up_scale", "down_scale"):
        assert len(weights[key]) == 3
        assert weights[key][1] is None, f"{key}[1] should be skipped for a GPU expert"
        assert weights[key][0] is not None and weights[key][2] is not None

    assert weights["gate"][0].shape == (128, 32)
    assert weights["gate_scale"][0].shape == (128, 4)
    assert weights["down"][0].shape == (64, 64)
    assert weights["down_scale"][0].shape == (64, 8)
    assert weights["gate"][0].dtype == torch.uint8
    assert weights["gate_scale"][0].dtype == torch.bfloat16

    # Without a mask every expert is loaded.
    full = loader.load_experts("model.layers.7")
    assert all(t is not None for t in full["gate"])

    with pytest.raises(ValueError, match="gpu_experts_mask length"):
        loader.load_experts("model.layers.7", gpu_experts_mask=torch.ones(5, dtype=torch.bool))


@pytest.mark.parametrize("global_suffix", ("weight_scale_2", None))
def test_nvfp4_fold_matches_fp64(tmp_path, global_suffix):
    make_checkpoint(tmp_path, global_suffix=global_suffix)
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    weights = loader.load_experts("model.layers.7")

    # The loader folds e4m3(block scale) * fp32(global) into one bf16 scale per
    # 16-K group. Against a float64 dequant of the same bytes the only error
    # allowed is bf16 rounding of that fold. An absent global scale (checkpoints
    # that baked it into the block scales) folds as a multiplication by 1.0.
    gate_pfx = f"{PREFIX}.0.{PROJ_NAMES[0]}"
    packed = weights["gate"][0]
    folded = weights["gate_scale"][0]
    rows, half_cols = packed.shape
    cols = half_cols * 2
    block_scale_ref = loader.load_tensor(f"{gate_pfx}.{BLOCK_SUFFIX}").to(torch.float64)
    if global_suffix is None:
        global_ref = torch.ones((), dtype=torch.float64)
    else:
        global_ref = loader.load_tensor(f"{gate_pfx}.{global_suffix}").to(torch.float64).reshape(())
    scale_ref = block_scale_ref * global_ref
    torch.testing.assert_close(folded.float().to(torch.float64), scale_ref.to(torch.bfloat16).to(torch.float64))

    torch.manual_seed(3)
    activation = torch.randn(cols, dtype=torch.float64)
    reference = unpack_fp4(packed, scale_ref.repeat_interleave(16, -1)) @ activation
    folded_out = unpack_fp4(packed, folded.double().repeat_interleave(16, -1)) @ activation
    cosine = F.cosine_similarity(reference, folded_out, dim=0)
    assert cosine >= 0.99999


@pytest.mark.parametrize("global_suffix", ("weight_scale_2", None))
def test_nvfp4_unfolded_scales_are_exact(tmp_path, global_suffix):
    """fold_global_scale=False must return the e4m3 block scales bit-exactly
    plus one fp32 global per expert, so a backend applying the global in fp32
    reproduces e4m3 * global with no bf16 rounding of the product."""
    make_checkpoint(tmp_path, global_suffix=global_suffix)
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    mask = torch.tensor([False, True, False])
    weights = loader.load_experts("model.layers.7", gpu_experts_mask=mask, fold_global_scale=False)

    for proj_idx, key in enumerate(("gate", "up", "down")):
        assert weights[f"{key}_global"][1] is None
        for expert in (0, 2):
            pfx = f"{PREFIX}.{expert}.{PROJ_NAMES[proj_idx]}"
            block = loader.load_tensor(f"{pfx}.{BLOCK_SUFFIX}").to(torch.float32)
            scale = weights[f"{key}_scale"][expert]
            assert scale.dtype == torch.bfloat16
            assert torch.equal(scale.float(), block)
            # The checkpoint stores the global as float32; an absent one is 1.0.
            expected_global = torch.tensor(0.0123, dtype=torch.float32).item() if global_suffix else 1.0
            assert weights[f"{key}_global"][expert] == expected_global
