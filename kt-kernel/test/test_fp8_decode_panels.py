"""The FP8 small-m GEMV (KT_FP8_DECODE_PANELS) must match float_mat_vec_kgroup bit for bit.

KT_FP8_DECODE_PANELS, KT_DECODE_FUSED and the other knobs are read once per process, so
each configuration runs in a worker subprocess on the same seeded block-wise FP8 weights
and inputs. The 4-subpool pool gives each node 384 gate/up columns (three 128-wide ranges,
cut into 32-column decode parts) and a 384-wide down K (three k groups: a two-group pair
and a tail group). Hidden 1152 keeps K within one K_BLOCK (prefetched stream, nine k
groups); hidden 7296 spans two K_BLOCKs (no prefetch). Prefill calls cover the vector path
for every expert (qlen 2, 5), the vector path inside mat_mul_kgroup for experts under 16
rows (qlen 12, skewed 9), and an expert with 40 rows (AMX, or two M blocks of the vector
kernel under KT_FP8_PREFILL_AMX=0).
"""

import os
import subprocess
import sys

import pytest
import torch

EXPERTS, TOP_K, INTERMEDIATE, CALLS, GROUP = 8, 4, 1536, 45, 128


def _worker(out_path, hidden):
    import kt_kernel

    ext = kt_kernel.kt_kernel_ext
    torch.manual_seed(11)

    def fp8_weights(n, k):
        # 0x7F / 0xFF are NaN to torch; the kernel's table maps them finite. Avoid both.
        codes = torch.randint(0, 256, (EXPERTS, n, k), dtype=torch.int32).to(torch.uint8)
        codes[(codes == 0x7F) | (codes == 0xFF)] = 0x7E
        scales = torch.rand(EXPERTS, n // GROUP, k // GROUP, dtype=torch.float32) * 0.0004 + 0.0001
        return codes.contiguous(), scales.contiguous()

    gate, gate_scales = fp8_weights(INTERMEDIATE, hidden)
    up, up_scales = fp8_weights(INTERMEDIATE, hidden)
    down, down_scales = fp8_weights(hidden, INTERMEDIATE)
    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(EXPERTS, TOP_K, hidden, INTERMEDIATE, 0)
    config.max_len = 64
    config.quant_config.bits = 8
    config.quant_config.group_size = GROUP
    config.quant_config.zero_point = False
    config.gate_proj, config.up_proj, config.down_proj = [w.data_ptr() for w in (gate, up, down)]
    config.gate_scale, config.up_scale, config.down_scale = [
        s.data_ptr() for s in (gate_scales, up_scales, down_scales)
    ]
    config.pool = pool.backend_
    moe = ext.moe.AMXFP8_MOE(config)
    mapping = torch.arange(EXPERTS, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()

    def forward(ids):
        qlen = ids.shape[0]
        weights = torch.rand(qlen, TOP_K, dtype=torch.float32).contiguous()
        inputs = (torch.randn(qlen, hidden) * 0.1).to(torch.bfloat16).contiguous()
        output = torch.empty(qlen, hidden, dtype=torch.bfloat16)
        batch = torch.tensor([qlen], dtype=torch.int32)
        pool.submit(
            moe.forward_task(
                batch.data_ptr(), TOP_K, ids.data_ptr(), weights.data_ptr(), inputs.data_ptr(), output.data_ptr(), False
            )
        )
        pool.sync()
        return output

    outputs = []
    for call in range(CALLS):
        # 0..TOP_K CPU routes; -1 stands for a route served elsewhere (skipped).
        cpu_routes = call % (TOP_K + 1)
        ids = torch.full((1, TOP_K), -1, dtype=torch.int64)
        ids[0, :cpu_routes] = torch.randperm(EXPERTS)[:cpu_routes]
        outputs.append(forward(ids[:, torch.randperm(TOP_K)].contiguous()))
    for qlen in (2, 5, 12):
        ids = torch.stack([torch.randperm(EXPERTS)[:TOP_K] for _ in range(qlen)]).contiguous()
        outputs.append(forward(ids))
    for qlen in (9, 40):
        outputs.append(forward(torch.randperm(EXPERTS)[:TOP_K].repeat(qlen, 1).contiguous()))
    torch.save(outputs, out_path)


def _run(tmp_path, hidden, name, **env):
    out = tmp_path / f"{hidden}-{name}.pt"
    worker_env = {k: v for k, v in os.environ.items() if not k.startswith("KT_")}
    worker_env.update(env)
    subprocess.run([sys.executable, __file__, str(out), str(hidden)], env=worker_env, check=True, timeout=600)
    return torch.load(out)


def _equal(a, b):
    return len(a) == len(b) and all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b))


@pytest.mark.cpu
@pytest.mark.parametrize("hidden", [1152, 7296])
def test_fp8_decode_panels_match_matvec(tmp_path, hidden):
    try:
        import kt_kernel

        if not hasattr(kt_kernel.kt_kernel_ext.moe, "AMXFP8_MOE"):
            pytest.skip("AMXFP8_MOE not compiled in")
    except ImportError:
        pytest.skip("kt_kernel not installed")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "avx512_bf16" not in cpuinfo.read():
            pytest.skip("AVX512-BF16 is unavailable")
    reference = _run(tmp_path, hidden, "matvec", KT_FP8_DECODE_PANELS="0", KT_DECODE_FUSED="0")
    for name, env in {
        "matvec_fused": dict(KT_FP8_DECODE_PANELS="0"),
        "panels_fused": {},
        "panels_phased": dict(KT_DECODE_FUSED="0"),
        "panels_no_prefetch": dict(KT_FP8_DECODE_PREFETCH="0"),
        "panels_combined_units": dict(KT_DECODE_SPLIT_GATE_UP_BELOW="0", KT_DECODE_SPLIT_GATE_UP_OCCUPANCY="0"),
    }.items():
        assert _equal(_run(tmp_path, hidden, name, **env), reference), name
    vector_reference = _run(tmp_path, hidden, "matvec_vector_prefill", KT_FP8_DECODE_PANELS="0", KT_FP8_PREFILL_AMX="0")
    vector = _run(tmp_path, hidden, "panels_vector_prefill", KT_FP8_PREFILL_AMX="0")
    assert _equal(vector, vector_reference), "panels_vector_prefill"


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]))
