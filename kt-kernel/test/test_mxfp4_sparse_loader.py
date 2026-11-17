"""MXFP4 loader: sparse GPU-expert skip, ue8m0 scale folding."""

import torch

from kt_kernel.utils.loader import MXFP4SafeTensorLoader

# DeepSeek-V4 naming under base_key "model.layers.1": the checkpoint omits the
# `model.` prefix and uses `.scale` instead of `.weight_scale`.
PREFIX, PROJ_NAMES, SCALE_SUFFIX = "layers.1.ffn.experts", ("w1", "w3", "w2"), "scale"


def make_fake(prefix, proj_names, scale_suffix, expert_count=3):
    loader = object.__new__(MXFP4SafeTensorLoader)
    loaded_keys = []
    gate = proj_names[0]

    def has_tensor(key):
        return any(
            key == f"{prefix}.{expert}.{gate}.weight" or key == f"{prefix}.{expert}.{gate}.{scale_suffix}"
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


def test_mxfp4_loader_skips_gpu_resident_experts():
    loader, loaded_keys = make_fake(PREFIX, PROJ_NAMES, SCALE_SUFFIX)

    weights = loader.load_experts("model.layers.1", gpu_experts_mask=torch.tensor([True, False, True]))

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
    assert set(k for k in loaded_keys if k.endswith(".weight")) == {f"{PREFIX}.1.{proj}.weight" for proj in PROJ_NAMES}

    # ue8m0 127 encodes 2^(127-127) = 1.0; lossless in bf16.
    assert weights["gate_scale"][1].dtype == torch.bfloat16
    torch.testing.assert_close(weights["gate_scale"][1], torch.full((2, 1), 1.0, dtype=torch.bfloat16))
