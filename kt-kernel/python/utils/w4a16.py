"""compressed-tensors pack-quantized int4 experts for the W4A16 kernels (int4-w4a16-moe.hpp).

The format llm-compressor writes for its W4A16 (symmetric) and W4A16_ASYM schemes, any
group size or the channel strategy, per expert: `weight_packed` (int32, eight codes
q + 8 per word along K, lowest nibble first), `weight_scale` [n, k / group] in BF16,
`weight_shape`, and with asymmetric schemes `weight_zero_point` (int32, eight codes
zp + 8 per word along n).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os

import torch

from .loader import MOE_NAMING_PROBE_ORDER, MOE_NAMING_VARIANTS, SafeTensorLoader, base_key_variants

logger = logging.getLogger(__name__)

PROJECTIONS = ("gate", "up", "down")


def check_checkpoint(weight_path: str) -> None:
    """Raise unless weight_path's config.json declares weight-only int4 that W4A16 reads."""
    with open(os.path.join(weight_path, "config.json")) as f:
        config = json.load(f)
    quant = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
    if not quant:
        raise ValueError(f"W4A16 needs a quantization_config in {weight_path}/config.json")
    if quant.get("quant_method") != "compressed-tensors":
        raise NotImplementedError(f"W4A16 does not read quant_method={quant.get('quant_method')!r} checkpoints")
    if quant.get("format") != "pack-quantized":
        raise NotImplementedError(
            f"W4A16 reads compressed-tensors format 'pack-quantized', got {quant.get('format')!r}"
        )
    int4_groups = [
        group
        for group in quant.get("config_groups", {}).values()
        if group.get("weights") and group["weights"].get("num_bits") == 4 and group["weights"].get("type") == "int"
    ]
    if not int4_groups:
        raise ValueError("W4A16: the checkpoint's compressed-tensors config has no int4 weight group")
    for group in int4_groups:
        weights = group["weights"]
        if group.get("input_activations"):
            raise NotImplementedError("W4A16 reads weight-only int4; this checkpoint also quantizes activations")
        if weights.get("strategy") not in ("group", "channel") or weights.get("dynamic"):
            raise NotImplementedError(
                f"W4A16 reads static group or channel scales, got strategy={weights.get('strategy')!r}"
            )
        if weights.get("actorder") == "group":
            raise NotImplementedError("W4A16 does not read actorder='group' (g_idx) checkpoints")


def unpack_zero_points(packed: torch.Tensor, n: int) -> torch.Tensor:
    """[ceil(n / 8), groups] int32 zero points to uint8 codes [n, groups]."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
    codes = (packed.to(torch.int32).unsqueeze(1) >> shifts) & 15
    return codes.reshape(-1, packed.shape[1])[:n].to(torch.uint8).contiguous()


class W4A16SafeTensorLoader(SafeTensorLoader):
    """Per-expert codes [n, k / 2] uint8, scales [n, groups] BF16 and zero codes [n, groups]
    uint8 (None for symmetric checkpoints), in checkpoint expert order."""

    def _experts_prefix(self, base_key: str) -> tuple[str, tuple[str, str, str]]:
        for variant in MOE_NAMING_PROBE_ORDER:
            path_template, proj_names = MOE_NAMING_VARIANTS[variant]
            for key in base_key_variants(base_key):
                prefix = path_template.format(base=key)
                if self.has_tensor(f"{prefix}.0.{proj_names[0]}.weight_packed"):
                    return prefix, proj_names
        raise ValueError(f"W4A16: no compressed-tensors experts ('...experts.0.<gate>.weight_packed') under {base_key}")

    def _projection(self, key: str, device: str):
        if self.has_tensor(f"{key}.weight_g_idx"):
            raise NotImplementedError(f"W4A16 does not read activation-ordered weights ({key}.weight_g_idx)")
        packed = self.load_tensor(f"{key}.weight_packed", device)
        scale = self.load_tensor(f"{key}.weight_scale", device).contiguous()
        n, k = packed.shape[0], packed.shape[1] * 8
        if self.has_tensor(f"{key}.weight_shape"):
            shape = tuple(int(v) for v in self.load_tensor(f"{key}.weight_shape", "cpu").tolist())
            if shape != (n, k):
                raise ValueError(f"W4A16: {key}.weight_packed {tuple(packed.shape)} does not hold weight_shape {shape}")
        if packed.dtype != torch.int32 or scale.dim() != 2 or scale.shape[0] != n or k % scale.shape[1]:
            raise ValueError(
                f"W4A16: {key} has weight_packed {packed.dtype} {tuple(packed.shape)} and weight_scale "
                f"{tuple(scale.shape)}; expected int32 [n, k / 8] and [n, k / group]"
            )
        if scale.dtype != torch.bfloat16:
            raise NotImplementedError(f"W4A16 reads BF16 scales; {key}.weight_scale is {scale.dtype}")
        zero_key = f"{key}.weight_zero_point"
        zero = unpack_zero_points(self.load_tensor(zero_key, device), n) if self.has_tensor(zero_key) else None
        if zero is not None and zero.shape != scale.shape:
            raise ValueError(f"W4A16: {zero_key} holds {tuple(zero.shape)} zero points for {tuple(scale.shape)} scales")
        return packed.contiguous().view(torch.uint8).view(n, k // 2), scale, zero

    def load_experts(self, base_key: str, gpu_experts_mask: torch.Tensor | None = None, device: str = "cpu"):
        prefix, proj_names = self._experts_prefix(base_key)
        count = 0
        while self.has_tensor(f"{prefix}.{count}.{proj_names[0]}.weight_packed"):
            count += 1
        skipped = [False] * count if gpu_experts_mask is None else gpu_experts_mask.to(torch.bool).tolist()
        if len(skipped) != count:
            raise ValueError(
                f"gpu_experts_mask length does not match the W4A16 expert count: {len(skipped)} != {count}"
            )
        result = {}
        for name, proj in zip(PROJECTIONS, proj_names):
            loaded = [
                (None, None, None) if skip else self._projection(f"{prefix}.{expert}.{proj}", device)
                for expert, skip in enumerate(skipped)
            ]
            result[name], result[f"{name}_scale"], result[f"{name}_zero"] = (list(column) for column in zip(*loaded))
        present = {
            zero is not None for name in PROJECTIONS for zero, skip in zip(result[f"{name}_zero"], skipped) if not skip
        }
        if len(present) > 1:
            raise ValueError(f"W4A16: some experts under {prefix} have zero points and some do not")
        return result


def part_group_size(group_sizes, k_per_part: int) -> int:
    """The largest group that every projection's groups and each NUMA part's share of
    down's K divide: groups may not span the parts TP_MOE splits the intermediate into."""
    group_size = math.gcd(*group_sizes, k_per_part)
    if group_size % 32:
        raise ValueError(
            f"W4A16 needs groups of a multiple of 32 within each NUMA part; groups {sorted(set(group_sizes))} "
            f"and {k_per_part} intermediate columns per part leave {group_size}"
        )
    return group_size


def regroup(tensors, group_size: int, target: int):
    """Per-group tensors [n, k / group_size] repeated to [n, k / target]; each weight keeps its
    scale and zero."""
    if group_size == target:
        return tensors
    repeat = group_size // target
    return [None if t is None else t.repeat_interleave(repeat, dim=1).contiguous() for t in tensors]


@functools.lru_cache(maxsize=None)
def log_regroup(group_sizes: tuple, k_per_part: int, target: int) -> None:
    logger.info("[KT] W4A16: groups %s over NUMA parts of %d columns; regrouped to %d", group_sizes, k_per_part, target)
