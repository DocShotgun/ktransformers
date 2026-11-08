"""MXFP8 experts served by the GPU are never read on the CPU.

The MXFP8 loader returns no tensors for GPU-resident experts, so their
per-expert pointers reach kt-kernel as 0. Loading must skip those experts, and
CPU-routed tokens must come out exactly as they do with every expert loaded.
"""

import pytest
import torch

import amx_backend
import kt_kernel

EXPERTS, TOPK, HIDDEN, INTERMEDIATE, GROUP = 4, 2, 512, 512, 32
GPU_EXPERTS = (0, 2)
BACKENDS = {"amx": "AMXMXFP8_KGroup_MOE", "avx2": "AVX2MXFP8_MOE"}


def _fp8_rows(rows, cols):
    weight = (torch.randn(EXPERTS, rows, cols) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.randint(124, 128, (EXPERTS, rows, cols // GROUP), dtype=torch.uint8)
    return weight.contiguous(), scale.contiguous()


def _run(backend):
    ext = kt_kernel.kt_kernel_ext
    torch.manual_seed(5)
    gate, up, down = _fp8_rows(INTERMEDIATE, HIDDEN), _fp8_rows(INTERMEDIATE, HIDDEN), _fp8_rows(HIDDEN, INTERMEDIATE)
    # Pinned like BaseMoEWrapper's mask: MOEConfig's 5th argument is ambiguous
    # (int num_gpu_experts vs mask pointer), and a pointer below 2**31 binds
    # as a count, silently leaving the mask unset.
    gpu_mask = torch.zeros(EXPERTS, dtype=torch.bool, pin_memory=True)
    gpu_mask[list(GPU_EXPERTS)] = True
    pool = ext.CPUInfer(8)
    moe_cls = getattr(ext.moe, BACKENDS[backend])

    def build(sparse):
        def pointers(tensor):
            return [[0 if sparse and gpu_mask[e] else tensor[e].data_ptr() for e in range(EXPERTS)]]

        config = ext.moe.MOEConfig(EXPERTS, TOPK, HIDDEN, INTERMEDIATE, gpu_mask.data_ptr())
        assert config.gpu_experts_mask == gpu_mask.data_ptr()
        assert config.num_gpu_experts == len(GPU_EXPERTS)
        config.max_len = 16
        config.gate_projs, config.up_projs, config.down_projs = (pointers(t[0]) for t in (gate, up, down))
        config.gate_scales, config.up_scales, config.down_scales = (pointers(t[1]) for t in (gate, up, down))
        config.quant_config.bits = 8
        config.quant_config.group_size = GROUP
        config.quant_config.zero_point = False
        config.pool = pool.backend_
        moe = moe_cls(config)
        mapping = torch.arange(EXPERTS, dtype=torch.int64)
        pool.submit(moe.load_weights_task(mapping.data_ptr()))
        pool.sync()
        return moe

    sparse, dense = build(sparse=True), build(sparse=False)

    qlen = 16
    ids = torch.stack([torch.randperm(EXPERTS)[:TOPK] for _ in range(qlen)]).to(torch.int64).contiguous()
    weights = torch.rand(qlen, TOPK, dtype=torch.float32).contiguous()
    inputs = (torch.randn(qlen, HIDDEN) * 0.1).to(torch.bfloat16).contiguous()

    def forward(moe, rows):
        batch_size = torch.tensor([rows], dtype=torch.int32)
        output = torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16)
        pool.submit(
            moe.forward_task(
                batch_size.data_ptr(), TOPK, ids.data_ptr(), weights.data_ptr(), inputs.data_ptr(), output.data_ptr(), False
            )
        )
        pool.sync()
        return output[:rows].clone()

    return {name: (forward(sparse, rows), forward(dense, rows)) for name, rows in (("prefill", qlen), ("decode", 1))}


def child_main():
    amx_backend.child_main(_run)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_mxfp8_skips_gpu_resident_experts(tmp_path, backend):
    if not hasattr(kt_kernel.kt_kernel_ext.moe, BACKENDS[backend]):
        pytest.skip(f"{BACKENDS[backend]} is not compiled in")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable for the pinned GPU-expert mask")
    if backend == "amx":
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            if "amx_bf16" not in cpuinfo.read():
                pytest.skip("AMX BF16 is unavailable")

    results = amx_backend.run_backend(__name__, backend, tmp_path)

    for name, (sparse, dense) in results.items():
        assert torch.isfinite(sparse.float()).all(), name
        assert sparse.float().abs().sum() > 0, name
        assert torch.equal(sparse, dense), name
