"""NVFP4 K-major scales kept as e4m3 codes must give exactly the outputs of the BF16 copy.

The block scales walk every positive e4m3 code (zero and the subnormals included). Each
setting of KT_NVFP4_E4M3_SCALES runs in its own process, since the loader reads it once.
"""

import os
import subprocess
import sys

import pytest
import torch

import kt_kernel

_RUN = r"""
import sys
import torch
import kt_kernel

ext = kt_kernel.kt_kernel_ext
experts, hidden, inter, topk = 4, 1024, 512, 2
gen = torch.Generator().manual_seed(3)
codes = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


def rows(n, k):
    packed = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, generator=gen)
    pick = torch.randint(0, codes.numel(), (experts, n, k // 16), generator=gen)
    return packed, codes[pick].to(torch.bfloat16).contiguous(), [2.0**-9] * experts


projections = [rows(inter, hidden), rows(inter, hidden), rows(hidden, inter)]
pool = ext.CPUInfer(8)
config = ext.moe.MOEConfig(experts, topk, hidden, inter, 0)
config.max_len = 64
config.gate_proj, config.up_proj, config.down_proj = [p[0].data_ptr() for p in projections]
config.gate_scale, config.up_scale, config.down_scale = [p[1].data_ptr() for p in projections]
config.gate_global_scales, config.up_global_scales, config.down_global_scales = [p[2] for p in projections]
config.quant_config.bits = 4
config.quant_config.group_size = 16
config.quant_config.zero_point = False
config.pool = pool.backend_
moe = ext.moe.AMXFP4_KGroup_MOE(config)
mapping = torch.arange(experts, dtype=torch.int64)
pool.submit(moe.load_weights_task(mapping.data_ptr()))
pool.sync()
outs = {}
for qlen in (1, 3, 4, 64):
    x = (torch.randn(qlen, hidden, generator=gen) * 0.1).to(torch.bfloat16).contiguous()
    ids = torch.stack([torch.randperm(experts, generator=gen)[:topk] for _ in range(qlen)]).contiguous()
    w = torch.rand(qlen, topk, generator=gen).contiguous()
    out = torch.zeros(qlen, hidden, dtype=torch.bfloat16)
    bsz = torch.tensor([qlen], dtype=torch.int32)
    pool.submit(moe.forward_task(bsz.data_ptr(), topk, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr(), False))
    pool.sync()
    outs[qlen] = out
torch.save(outs, sys.argv[1])
"""


def test_e4m3_kmajor_scales_match_bf16(tmp_path):
    if not hasattr(kt_kernel.kt_kernel_ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX FP4 extension is unavailable")
    outs = {}
    for flag in ("0", "1"):
        path = tmp_path / f"e4m3_{flag}.pt"
        env = dict(os.environ, KT_NVFP4_E4M3_SCALES=flag, KT_NVFP4_KMAJOR_WEIGHTS="1")
        subprocess.run([sys.executable, "-c", _RUN, str(path)], env=env, check=True, capture_output=True)
        outs[flag] = torch.load(path)
    for qlen, bf16 in outs["0"].items():
        assert bf16.abs().sum() > 0
        assert torch.equal(outs["1"][qlen].view(torch.int16), bf16.view(torch.int16)), f"qlen {qlen} differs"
