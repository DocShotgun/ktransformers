"""CPU buffer tuples handed out during a CUDA-graph capture must outlive it.

A captured MoE forward bakes the tuple's pinned and GPU pointers into the graph.
get_buffer() keeps one temp tuple and drops it when the batch size changes, so a
tuple captured at a size outside capture_bs (sglang's decoder tail graphs
capture 128..2048 rows) was freed by the next eager forward at another size,
and later replays read and wrote freed memory.
"""

import pytest
import torch

from kt_kernel.experts_base import KExpertsCPUBuffer

HIDDEN, TOPK = 64, 2


@pytest.fixture
def clean_buffers():
    saved = KExpertsCPUBuffer.capture_bs
    KExpertsCPUBuffer.capture_bs = [1]
    KExpertsCPUBuffer.capture_buffers.clear()
    KExpertsCPUBuffer.temp_bs = 0
    KExpertsCPUBuffer.temp_buffer = tuple()
    yield
    KExpertsCPUBuffer.capture_bs = saved
    KExpertsCPUBuffer.capture_buffers.clear()
    KExpertsCPUBuffer.temp_bs = 0
    KExpertsCPUBuffer.temp_buffer = tuple()


def _get(rows):
    x = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    return KExpertsCPUBuffer.get_buffer(x, TOPK)


def test_buffers_taken_during_capture_survive_size_changes(clean_buffers):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    eager_128 = _get(128)  # becomes the temp tuple before capture
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    scratch = torch.zeros(1, device="cuda")
    with torch.cuda.graph(graph, stream=stream):
        reused = _get(128)  # temp tuple reused inside the capture
        fresh = _get(256)  # new tuple allocated inside the capture
        scratch.add_(1)
    assert reused is eager_128

    _get(8192)  # eager forward at another size replaces the temp tuple
    assert _get(128) is reused
    assert _get(256) is fresh


def test_eager_size_changes_still_drop_uncaptured_tuples(clean_buffers):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _get(128)
    _get(8192)
    assert 128 not in KExpertsCPUBuffer.capture_buffers
    assert 8192 not in KExpertsCPUBuffer.capture_buffers
