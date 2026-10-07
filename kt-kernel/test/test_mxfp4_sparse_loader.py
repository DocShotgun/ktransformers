"""MXFP4 loader: naming variants, sparse GPU-expert skip, ue8m0 scale folding."""

import pytest
import torch

from kt_kernel.utils.loader import MXFP4SafeTensorLoader

# variant: (experts prefix under base_key "model.layers.1", (gate, up, down), scale suffix).
# deepseek_v4 checkpoints omit the `model.` prefix and use `.scale` instead of
# `.weight_scale`; the loader probes both forms per layer.
NAMINGS = {
    "hf_mlp": (
        "model.layers.1.mlp.experts",
        ("gate_proj", "up_proj", "down_proj"),
        "weight_scale",
    ),
    "deepseek_v4": ("layers.1.ffn.experts", ("w1", "w3", "w2"), "scale"),
    "mixtral": (
        "model.layers.1.block_sparse_moe.experts",
        ("w1", "w3", "w2"),
        "weight_scale",
    ),
    "mistral": ("model.layers.1.experts", ("w1", "w3", "w2"), "weight_scale"),
}


def make_fake(prefix, proj_names, scale_suffix, expert_count=3):
    loader = object.__new__(MXFP4SafeTensorLoader)
    loaded_keys = []
    gate = proj_names[0]

    def has_tensor(key):
        return any(
            key == f"{prefix}.{expert}.{gate}.weight"
            or key == f"{prefix}.{expert}.{gate}.{scale_suffix}"
            for expert in range(expert_count)
        )

    def load_tensor(key, device="cpu"):
        loaded_keys.append(key)
        if key.endswith(f".{scale_suffix}"):
            return torch.full((2, 1), 127, dtype=torch.uint8, device=device)
        return torch.zeros((2, 16), dtype=torch.uint8, device=device)

    loader.has_tensor = has_tensor
    loader.load_tensor = load_tensor
    return loader, loaded_keys


@pytest.mark.parametrize("variant", sorted(NAMINGS))
def test_mxfp4_loader_skips_gpu_resident_experts(variant):
    prefix, proj_names, scale_suffix = NAMINGS[variant]
    loader, loaded_keys = make_fake(prefix, proj_names, scale_suffix)

    weights = loader.load_experts(
        "model.layers.1", gpu_experts_mask=torch.tensor([True, False, True])
    )

    for projection in (
        "gate",
        "up",
        "down",
        "gate_scale",
        "up_scale",
        "down_scale",
    ):
        assert weights[projection][0] is None
        assert weights[projection][1] is not None
        assert weights[projection][2] is None
    # Only expert 1's weights are read (scale reads include resolver dtype probes).
    assert set(k for k in loaded_keys if k.endswith(".weight")) == {
        f"{prefix}.1.{proj}.weight" for proj in proj_names
    }

    # ue8m0 127 encodes 2^(127-127) = 1.0; lossless in bf16.
    assert weights["gate_scale"][1].dtype == torch.bfloat16
    torch.testing.assert_close(
        weights["gate_scale"][1], torch.full((2, 1), 1.0, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("variant", sorted(NAMINGS))
def test_mxfp4_loader_loads_all_experts_without_mask(variant):
    prefix, proj_names, scale_suffix = NAMINGS[variant]
    loader, _ = make_fake(prefix, proj_names, scale_suffix)

    weights = loader.load_experts("model.layers.1")
    assert all(t is not None for t in weights["gate"])
    assert weights["gate"][0].shape == (2, 16)
    assert weights["gate_scale"][0].shape == (2, 1)


def test_mxfp4_gpu_mask_length_mismatch():
    prefix, proj_names, scale_suffix = NAMINGS["hf_mlp"]
    loader, _ = make_fake(prefix, proj_names, scale_suffix)

    with pytest.raises(ValueError, match="gpu_experts_mask length"):
        loader.load_experts(
            "model.layers.1", gpu_experts_mask=torch.ones(5, dtype=torch.bool)
        )


def test_mxfp4_unresolved_layout_lists_probed_prefixes():
    loader = object.__new__(MXFP4SafeTensorLoader)
    loader.has_tensor = lambda key: False
    loader.load_tensor = lambda key, device="cpu": torch.zeros(1)

    with pytest.raises(ValueError) as excinfo:
        loader.load_experts("model.layers.1")
    message = str(excinfo.value)
    assert "model.layers.1.mlp.experts" in message
    assert "KT_MOE_NAMING" in message