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


def _stage(stream, hidden, ids, weights, host, bell=None, bell_value=0):
    ext.stage_moe_inputs(
        stream.cuda_stream, hidden.data_ptr(), hidden.numel() * 2, ids.data_ptr(), ids.dtype == torch.int32,
        weights.data_ptr(), ids.numel(), host[0].data_ptr(), host[1].data_ptr(), host[2].data_ptr(),
        0 if bell is None else bell.data_ptr(), bell_value,
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


def _ring_and_check(stream, hidden, ids, weights, host, bell, value, launch):
    """Start the staging behind a GPU delay, spin on the doorbell, and check that every
    staged byte is already visible when it flips."""
    expected = (hidden.cpu(), ids.to(torch.long).cpu(), weights.cpu())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        launch()
    while int(bell[0]) != value:
        pass
    assert torch.equal(host[0].view(torch.int16), expected[0].view(torch.int16))
    assert torch.equal(host[1], expected[1])
    assert torch.equal(host[2].view(torch.int32), expected[2].view(torch.int32))
    stream.synchronize()


def test_stage_rings_its_doorbell_after_every_store():
    """Given a doorbell word, the kernel (64 blocks at 64 rows) stores bell_value once,
    after every block's stores are visible to the host, eagerly and on graph replay."""
    _skip()
    rows = 64
    hidden, ids, weights = _inputs(rows, torch.int32, seed=11)
    host = _host_buffers(rows)
    bell = torch.zeros(1, dtype=torch.int32).pin_memory()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for value in (7, 8, 9):
        bell.zero_()
        host[0].zero_(), host[1].zero_(), host[2].zero_()
        _ring_and_check(stream, hidden, ids, weights, host, bell, value,
                        lambda: _stage(stream, hidden, ids, weights, host, bell, value))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _stage(stream, hidden, ids, weights, host, bell, 5)
    for seed in range(12, 16):
        new = _inputs(rows, torch.int32, seed=seed)
        hidden.copy_(new[0]), ids.copy_(new[1]), weights.copy_(new[2])
        torch.cuda.synchronize()
        bell.zero_()
        _ring_and_check(stream, hidden, ids, weights, host, bell, 5, graph.replay)
