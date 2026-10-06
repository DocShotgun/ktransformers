import torch

from kt_kernel.utils.loader import MXFP4SafeTensorLoader


def test_mxfp4_loader_skips_gpu_resident_experts():
    loader = object.__new__(MXFP4SafeTensorLoader)
    prefix = "model.layers.1.mlp.experts"
    loaded_keys = []

    def has_tensor(key):
        return any(
            key == f"{prefix}.{expert}.gate_proj.weight" for expert in range(3)
        )

    def load_tensor(key, device="cpu"):
        loaded_keys.append(key)
        if key.endswith(".weight_scale"):
            return torch.full((2, 1), 127, dtype=torch.uint8, device=device)
        return torch.zeros((2, 16), dtype=torch.uint8, device=device)

    loader.has_tensor = has_tensor
    loader.load_tensor = load_tensor

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
    assert loaded_keys
    assert all(f"{prefix}.1." in key for key in loaded_keys)
