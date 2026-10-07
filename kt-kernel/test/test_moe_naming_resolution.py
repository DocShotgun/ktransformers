"""MoE tensor-naming resolution: naming variants, base-key forms, quant families."""

import pytest
import torch

from kt_kernel.utils.loader import (
    MOE_NAMING_ENV_VAR,
    MOE_NAMING_PROBE_ORDER,
    MOE_NAMING_VARIANTS,
    base_key_variants,
    naming_variant_candidates,
    resolve_experts_layout,
)


class FakeLoader:
    """Dict-backed stand-in for SafeTensorLoader probing."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.loaded_keys = []

    def has_tensor(self, key):
        return key in self.tensors

    def load_tensor(self, key, device="cpu"):
        self.loaded_keys.append(key)
        return self.tensors[key]


def add_experts(
    tensors,
    prefix,
    proj_names,
    block_suffix,
    global_suffix=None,
    expert_count=2,
    scale_dtype=torch.uint8,
):
    for i in range(expert_count):
        for proj in proj_names:
            tensors[f"{prefix}.{i}.{proj}.weight"] = torch.zeros(2, 16, dtype=torch.uint8)
        gate = f"{prefix}.0.{proj_names[0]}"
        tensors[f"{gate}.{block_suffix}"] = torch.ones(2, 1, dtype=scale_dtype)
        if global_suffix is not None:
            tensors[f"{gate}.{global_suffix}"] = torch.tensor(0.5, dtype=torch.float32)
    return tensors


def test_naming_variant_candidates_default_order():
    assert naming_variant_candidates() == MOE_NAMING_PROBE_ORDER
    assert set(MOE_NAMING_PROBE_ORDER) == set(MOE_NAMING_VARIANTS)


def test_naming_variant_candidates_env_override(monkeypatch):
    monkeypatch.setenv(MOE_NAMING_ENV_VAR, "deepseek_v4")
    assert naming_variant_candidates() == ("deepseek_v4",)


def test_naming_variant_candidates_env_invalid(monkeypatch):
    monkeypatch.setenv(MOE_NAMING_ENV_VAR, "bogus")
    with pytest.raises(ValueError, match=MOE_NAMING_ENV_VAR):
        naming_variant_candidates()


def test_base_key_variants_strip_and_add():
    assert base_key_variants("model.layers.3") == [
        "model.layers.3",
        "layers.3",
        "model.language_model.layers.3",
        "language_model.model.layers.3",
    ]
    assert base_key_variants("layers.3") == [
        "layers.3",
        "model.language_model.layers.3",
        "language_model.model.layers.3",
    ]
    assert base_key_variants("model.language_model.layers.3") == [
        "model.language_model.layers.3",
        "layers.3",
        "language_model.model.layers.3",
    ]


@pytest.mark.parametrize(
    "variant, prefix, proj_names, block_suffix",
    [
        ("hf_mlp", "model.layers.3.mlp.experts", ("gate_proj", "up_proj", "down_proj"), "weight_scale"),
        ("deepseek_v4", "layers.3.ffn.experts", ("w1", "w3", "w2"), "scale"),
        ("mixtral", "model.layers.3.block_sparse_moe.experts", ("w1", "w3", "w2"), "weight_scale"),
        ("mistral", "model.layers.3.experts", ("w1", "w3", "w2"), "weight_scale"),
    ],
)
def test_mxfp4_naming_variants(variant, prefix, proj_names, block_suffix):
    loader = FakeLoader(add_experts({}, prefix, proj_names, block_suffix))
    layout = resolve_experts_layout(loader, "model.layers.3", family="mxfp4")
    assert layout is not None
    assert layout.variant == variant
    assert layout.prefix == prefix
    assert layout.family == "mxfp4"
    assert layout.block_scale_suffix == block_suffix
    assert layout.global_scale_suffix is None
    assert layout.weight_key(1, 0) == f"{prefix}.1.{proj_names[0]}.weight"
    assert layout.block_scale_key(1, 2) == f"{prefix}.1.{proj_names[2]}.{block_suffix}"
    assert layout.global_scale_key(1, 0) is None


@pytest.mark.parametrize(
    "global_suffix", ("weight_scale_2", "scaling_factor", "scale_2")
)
def test_nvfp4_global_scale_aliases(global_suffix):
    loader = FakeLoader(
        add_experts(
            {},
            "model.layers.3.mlp.experts",
            ("gate_proj", "up_proj", "down_proj"),
            "weight_scale",
            global_suffix=global_suffix,
            scale_dtype=torch.float8_e4m3fn,
        )
    )
    layout = resolve_experts_layout(loader, "model.layers.3")
    assert layout.family == "nvfp4"
    assert layout.global_scale_suffix == global_suffix
    assert layout.global_scale_key(0, 1) == (
        f"model.layers.3.mlp.experts.0.up_proj.{global_suffix}"
    )


@pytest.mark.parametrize(
    "scale_dtype, family",
    [
        (torch.uint8, "mxfp4"),
        (torch.float8_e4m3fn, "nvfp4"),
    ],
)
def test_bare_block_scale_dtype_disambiguation(scale_dtype, family):
    # Without a global-scale key, MXFP4 ue8m0 block scales and NVFP4 fp8 block
    # scales with a baked-in global are distinguished by scale dtype.
    loader = FakeLoader(
        add_experts(
            {},
            "model.layers.3.mlp.experts",
            ("gate_proj", "up_proj", "down_proj"),
            "weight_scale",
            scale_dtype=scale_dtype,
        )
    )
    layout = resolve_experts_layout(loader, "model.layers.3")
    assert layout.family == family


@pytest.mark.parametrize("block_suffix", ("weight_scale_inv", "scale_inv"))
def test_mxfp8_block_scale_suffixes(block_suffix):
    loader = FakeLoader(
        add_experts(
            {},
            "language_model.model.layers.3.block_sparse_moe.experts",
            ("w1", "w3", "w2"),
            block_suffix,
        )
    )
    layout = resolve_experts_layout(loader, "model.layers.3")
    assert layout.family == "mxfp8"
    assert layout.prefix == "language_model.model.layers.3.block_sparse_moe.experts"


def test_vl_prefixed_layers_resolve():
    # GLM/Qwen-VL checkpoints nest experts under model.language_model.layers.{L}.
    loader = FakeLoader(
        add_experts(
            {},
            "model.language_model.layers.3.mlp.experts",
            ("gate_proj", "up_proj", "down_proj"),
            "weight_scale",
        )
    )
    layout = resolve_experts_layout(loader, "model.layers.3", family="mxfp4")
    assert layout is not None
    assert layout.prefix == "model.language_model.layers.3.mlp.experts"


def test_family_filter_mismatch_returns_none():
    loader = FakeLoader(
        add_experts(
            {},
            "model.layers.3.mlp.experts",
            ("gate_proj", "up_proj", "down_proj"),
            "weight_scale",
        )
    )
    assert resolve_experts_layout(loader, "model.layers.3", family="nvfp4") is None
    assert resolve_experts_layout(loader, "model.layers.3", family="mxfp4") is not None


def test_env_override_prefers_named_variant(monkeypatch):
    # A checkpoint carrying both layouts resolves hf_mlp by default and the
    # pinned variant under KT_MOE_NAMING.
    tensors = add_experts(
        {},
        "model.layers.3.mlp.experts",
        ("gate_proj", "up_proj", "down_proj"),
        "weight_scale",
    )
    add_experts(tensors, "model.layers.3.ffn.experts", ("w1", "w3", "w2"), "scale")
    loader = FakeLoader(tensors)

    assert resolve_experts_layout(loader, "model.layers.3").variant == "hf_mlp"
    monkeypatch.setenv(MOE_NAMING_ENV_VAR, "deepseek_v4")
    assert resolve_experts_layout(loader, "model.layers.3").variant == "deepseek_v4"


def test_unresolved_layout_returns_none():
    loader = FakeLoader({})
    assert resolve_experts_layout(loader, "model.layers.3") is None