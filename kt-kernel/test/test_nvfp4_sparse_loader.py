"""NVFP4 loader: sparse GPU-expert skip and BF16 scale folding."""

import pytest
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    pytest.skip("safetensors is unavailable", allow_module_level=True)

from kt_kernel.utils.loader import NVFP4SafeTensorLoader

PROJS = ("gate_proj", "up_proj", "down_proj")
FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def make_checkpoint(tmp_path, expert_count=3, hidden=64, intermediate=128):
    tensors = {}
    for i in range(expert_count):
        for proj in PROJS:
            rows, cols = (
                (intermediate, hidden)
                if proj != "down_proj"
                else (hidden, intermediate)
            )
            prefix = f"model.layers.7.mlp.experts.{i}.{proj}"
            tensors[f"{prefix}.weight"] = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8
            )
            tensors[f"{prefix}.weight_scale"] = (
                torch.randn(rows, cols // 16).clamp(-4, 4).to(torch.float8_e4m3fn)
            )
            tensors[f"{prefix}.weight_scale_2"] = torch.tensor(
                0.0123, dtype=torch.float32
            )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def unpack_fp4(packed, scales):
    values = torch.tensor(FP4_VALUES, dtype=torch.float64)
    unscaled = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
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


def test_nvfp4_fold_matches_fp64(tmp_path):
    make_checkpoint(tmp_path)
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    weights = loader.load_experts("model.layers.7")

    # The loader folds e4m3(block scale) * fp32(global) into one bf16 scale per
    # 16-K group. Against a float64 dequant of the same bytes the only error
    # allowed is bf16 rounding of that fold.
    packed = weights["gate"][0]
    folded = weights["gate_scale"][0]
    rows, half_cols = packed.shape
    cols = half_cols * 2
    block_scale_ref = (
        loader.load_tensor("model.layers.7.mlp.experts.0.gate_proj.weight_scale")
        .to(torch.float64)
    )
    global_ref = loader.load_tensor(
        "model.layers.7.mlp.experts.0.gate_proj.weight_scale_2"
    ).to(torch.float64).reshape(())
    scale_ref = block_scale_ref * global_ref
    torch.testing.assert_close(folded.float().to(torch.float64), scale_ref.to(torch.bfloat16).to(torch.float64))

    torch.manual_seed(3)
    activation = torch.randn(cols, dtype=torch.float64)
    reference = unpack_fp4(packed, scale_ref.repeat_interleave(16, -1)) @ activation
    folded_out = unpack_fp4(
        packed, folded.double().repeat_interleave(16, -1)
    ) @ activation
    cosine = F.cosine_similarity(reference, folded_out, dim=0)
    assert cosine >= 0.99999