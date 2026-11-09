"""stage_moe_inputs must write what the three copy_ calls (and the int64 cast) wrote."""

import pytest
import torch

import kt_kernel

ext = kt_kernel.kt_kernel_ext
HIDDEN, TOPK = 5120, 6


def _inputs(rows, ids_dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16).cuda()
    ids = torch.randint(-1, 384, (rows, TOPK), generator=generator).to(ids_dtype).cuda()
    weights = torch.rand(rows, TOPK, generator=generator).cuda()
    return hidden, ids, weights


def _host_buffers(rows):
    return (
        torch.empty(rows, HIDDEN, dtype=torch.bfloat16).pin_memory(),
        torch.empty(rows, TOPK, dtype=torch.int64).pin_memory(),
        torch.empty(rows, TOPK, dtype=torch.float32).pin_memory(),
    )


def _stage(stream, hidden, ids, weights, host):
    ext.stage_moe_inputs(
        stream.cuda_stream, hidden.data_ptr(), hidden.numel() * 2, ids.data_ptr(), ids.dtype == torch.int32,
        weights.data_ptr(), ids.numel(), host[0].data_ptr(), host[1].data_ptr(), host[2].data_ptr(),
    )


def _assert_matches(host, hidden, ids, weights):
    assert torch.equal(host[0].view(torch.int16), hidden.cpu().view(torch.int16))
    assert torch.equal(host[1], ids.to(torch.long).cpu())
    assert torch.equal(host[2].view(torch.int32), weights.cpu().view(torch.int32))


def _skip():
    if not torch.cuda.is_available() or not hasattr(ext, "stage_moe_inputs"):
        pytest.skip("kt-kernel was built without CUDA")


@pytest.mark.parametrize("rows", [1, 4, 64])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_stage_matches_copies(rows, ids_dtype):
    _skip()
    hidden, ids, weights = _inputs(rows, ids_dtype, seed=rows)
    host = _host_buffers(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _stage(stream, hidden, ids, weights, host)
    stream.synchronize()
    _assert_matches(host, hidden, ids, weights)


def test_stage_replays_in_a_graph():
    _skip()
    hidden, ids, weights = _inputs(1, torch.int32, seed=0)
    host = _host_buffers(1)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _stage(stream, hidden, ids, weights, host)
    for seed in range(1, 6):
        new = _inputs(1, torch.int32, seed=seed)
        hidden.copy_(new[0]), ids.copy_(new[1]), weights.copy_(new[2])
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        _assert_matches(host, hidden, ids, weights)
