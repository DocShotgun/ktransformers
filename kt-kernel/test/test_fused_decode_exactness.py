"""The fused single-dispatch MoE decode must match the phased decode bit for bit.

KT_DECODE_FUSED and the FP4 decode widths are read once per process, so each
configuration runs in a worker subprocess on the same seeded MXFP4 weights and
decode inputs. The 4-subpool pool gives 256-row gate/up slices, so widths 64 and
32 cover both fused unit layouts (gate and up split, and combined).
"""

import os
import subprocess
import sys

import pytest
import torch

EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, CALLS = 8, 4, 2048, 1024, 40


def _worker(out_path):
    import kt_kernel

    ext = kt_kernel.kt_kernel_ext
    torch.manual_seed(7)
    packed = [
        torch.randint(0, 256, (EXPERTS, INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8).contiguous(),
        torch.randint(0, 256, (EXPERTS, INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8).contiguous(),
        torch.randint(0, 256, (EXPERTS, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8).contiguous(),
    ]
    scales = [
        (2.0 ** torch.randint(-6, -3, (EXPERTS, INTERMEDIATE, HIDDEN // 32))).to(torch.bfloat16).contiguous(),
        (2.0 ** torch.randint(-6, -3, (EXPERTS, INTERMEDIATE, HIDDEN // 32))).to(torch.bfloat16).contiguous(),
        (2.0 ** torch.randint(-6, -3, (EXPERTS, HIDDEN, INTERMEDIATE // 32))).to(torch.bfloat16).contiguous(),
    ]
    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, 0)
    config.max_len = 16
    config.gate_proj, config.up_proj, config.down_proj = [w.data_ptr() for w in packed]
    config.gate_scale, config.up_scale, config.down_scale = [s.data_ptr() for s in scales]
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.pool = pool.backend_
    moe = ext.moe.AMXFP4_KGroup_MOE(config)
    mapping = torch.arange(EXPERTS, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    outputs = []
    batch = torch.tensor([1], dtype=torch.int32)
    for call in range(CALLS):
        # 0..TOP_K CPU routes; -1 stands for a route served elsewhere (skipped).
        cpu_routes = call % (TOP_K + 1)
        ids = torch.full((1, TOP_K), -1, dtype=torch.int64)
        ids[0, :cpu_routes] = torch.randperm(EXPERTS)[:cpu_routes]
        ids = ids[:, torch.randperm(TOP_K)].contiguous()
        weights = torch.rand(1, TOP_K, dtype=torch.float32).contiguous()
        inputs = (torch.randn(1, HIDDEN) * 0.1).to(torch.bfloat16).contiguous()
        output = torch.empty(1, HIDDEN, dtype=torch.bfloat16)
        pool.submit(
            moe.forward_task(
                batch.data_ptr(), TOP_K, ids.data_ptr(), weights.data_ptr(), inputs.data_ptr(), output.data_ptr(), False
            )
        )
        pool.sync()
        outputs.append(output)
    torch.save(torch.stack(outputs), out_path)


def _run(tmp_path, name, **env):
    out = tmp_path / f"{name}.pt"
    worker_env = {k: v for k, v in os.environ.items() if not k.startswith("KT_")}
    worker_env.update(env)
    subprocess.run([sys.executable, __file__, str(out)], env=worker_env, check=True, timeout=600)
    return torch.load(out)


@pytest.mark.cpu
def test_fused_decode_matches_phased(tmp_path):
    try:
        import kt_kernel

        if not hasattr(kt_kernel.kt_kernel_ext.moe, "AMXFP4_KGroup_MOE"):
            pytest.skip("AMXFP4_KGroup_MOE not compiled in")
    except ImportError:
        pytest.skip("kt_kernel not installed")
    reference = _run(tmp_path, "phased", KT_DECODE_FUSED="0", KT_FP4_DECODE_WIDTH="64")
    for name, env in {
        "fused_split_units": dict(KT_DECODE_FUSED="1", KT_FP4_DECODE_WIDTH="64"),
        "fused_combined_units": dict(KT_DECODE_FUSED="1", KT_FP4_DECODE_WIDTH="32"),
        "fused_forced_split": dict(KT_DECODE_FUSED="1", KT_FP4_DECODE_WIDTH="32", KT_DECODE_SPLIT_GATE_UP_BELOW="64"),
        "fused_narrow_down": dict(KT_DECODE_FUSED="1", KT_FP4_DECODE_DOWN_WIDTH="64"),
    }.items():
        result = _run(tmp_path, name, **env)
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16)), name


if __name__ == "__main__":
    _worker(sys.argv[1])
