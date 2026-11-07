"""CUDA-graph replay of a stream-submitted MoE forward.

A captured `submit_with_cuda_stream` / `sync_with_cuda_stream` pair re-runs its
host callbacks with the same argument blocks on every replay. Each replay must
recompute the MoE on the current pinned inputs; freeing or reusing those blocks
after the first callback turns later replays into use-after-free (crash or
garbage output).
"""

import pytest
import torch

import kt_kernel
from test_nvfp4_amx_prefill import make_nvfp4_rows

ext = kt_kernel.kt_kernel_ext


def _build_moe(pool, experts, topk, hidden, intermediate, max_len):
    torch.manual_seed(5)
    projections = [
        make_nvfp4_rows(experts, intermediate, hidden),
        make_nvfp4_rows(experts, intermediate, hidden),
        make_nvfp4_rows(experts, hidden, intermediate),
    ]
    config = ext.moe.MOEConfig(experts, topk, hidden, intermediate, 0)
    config.max_len = max_len
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
    return moe, projections


def test_graph_replays_recompute_moe_on_current_inputs():
    if not torch.cuda.is_available() or not hasattr(ext, "CPUInfer"):
        pytest.skip("CUDA is unavailable")
    if not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX FP4 extension is unavailable")
    if not hasattr(ext.CPUInfer, "submit_with_cuda_stream"):
        pytest.skip("kt-kernel was built without CUDA host callbacks")

    experts, topk, hidden, intermediate, qlen = 4, 2, 256, 256, 4
    pool = ext.CPUInfer(8)
    moe, _ = _build_moe(pool, experts, topk, hidden, intermediate, qlen)

    ids = torch.zeros(qlen, topk, dtype=torch.int64).pin_memory()
    weights = torch.zeros(qlen, topk, dtype=torch.float32).pin_memory()
    inputs = torch.zeros(qlen, hidden, dtype=torch.bfloat16).pin_memory()
    output = torch.zeros(qlen, hidden, dtype=torch.bfloat16).pin_memory()
    batch_size = torch.tensor([qlen], dtype=torch.int32).pin_memory()

    def task():
        return moe.forward_task(
            batch_size.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(),
            inputs.data_ptr(), output.data_ptr(), False,
        )

    def fill(seed):
        generator = torch.Generator().manual_seed(seed)
        ids.copy_(torch.stack([torch.randperm(experts, generator=generator)[:topk] for _ in range(qlen)]))
        weights.copy_(torch.rand(qlen, topk, generator=generator))
        inputs.copy_((torch.randn(qlen, hidden, generator=generator) * 0.1).to(torch.bfloat16))

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    fill(0)
    with torch.cuda.stream(stream):
        pool.submit_with_cuda_stream(stream.cuda_stream, task())  # warm-up, eager
        pool.sync_with_cuda_stream(stream.cuda_stream, 0)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        pool.submit_with_cuda_stream(stream.cuda_stream, task())
        pool.sync_with_cuda_stream(stream.cuda_stream, 0)

    for replay in range(20):
        fill(replay + 1)
        output.zero_()
        # replay() launches on the current stream; the captured sync callback
        # is what makes the stream wait for the CPU forward.
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        replayed = output.clone()

        output.zero_()
        pool.submit(task())
        pool.sync()
        assert replayed.abs().sum() > 0, f"replay {replay} produced no output"
        assert torch.equal(replayed, output), f"replay {replay} differs from an eager forward"
