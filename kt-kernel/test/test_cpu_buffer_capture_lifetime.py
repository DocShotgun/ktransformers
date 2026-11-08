"""KT's CPU transfer buffers: one arena that every batch size views.

Captured graphs bake the arena's pointers and CPU tasks run after the host has
moved on, so nothing may be freed while either can still use it: every row
count is a view of one arena, growth keeps the outgrown arena, and the per-size
qlen scalars stay put. Graphs captured at one size must keep working while
eager forwards at other sizes reuse the same memory.
"""

import pytest
import torch

import amx_backend
import kt_kernel
from kt_kernel.experts_base import KExpertsCPUBuffer

HIDDEN, TOPK = 64, 2


@pytest.fixture
def clean_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    KExpertsCPUBuffer.clear()
    yield
    KExpertsCPUBuffer.clear()


def _get(rows):
    x = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    return KExpertsCPUBuffer.get_buffer(x, TOPK)


def _pointers(views):
    # Field 5 holds the per-size qlen scalars, which are not arena views.
    return [[slot.data_ptr() for slot in field] for index, field in enumerate(views) if index != 5]


def test_every_row_count_views_one_arena(clean_buffers):
    KExpertsCPUBuffer.reserve(8192, HIDDEN, TOPK, device=torch.device("cuda", torch.cuda.current_device()))
    small, large = _get(128), _get(8192)
    assert _pointers(small) == _pointers(large)
    assert small[0][0].shape == (128, HIDDEN)
    assert large[6][1].shape == (8192, HIDDEN)


def test_growth_keeps_the_outgrown_arena(clean_buffers):
    first = _get(128)[0][0].data_ptr()
    grown = _get(129)
    assert grown[0][0].data_ptr() != first
    # At least doubling bounds what callers that never reserve leave behind.
    assert KExpertsCPUBuffer._host_arenas[(HIDDEN, TOPK)]["max_rows"] == 256
    assert any(arena["input"][0].data_ptr() == first for arena in KExpertsCPUBuffer._retired)


def test_qlen_scalars_stay_put_per_row_count(clean_buffers):
    before = _get(5)[5]
    _get(9)  # grows the arena and drops the cached views
    after = _get(5)[5]
    assert [t.data_ptr() for t in after] == [t.data_ptr() for t in before]
    assert [int(t.item()) for t in after] == [5] * KExpertsCPUBuffer.buffer_depth


def _run(backend, weight_dir):
    """Wrapper forwards in a child process: graphs at 3 and 5 rows replayed
    between eager forwards at 4 and 64 rows, each checked against a forward that
    bypasses the arena."""
    from kt_kernel import KTMoEWrapper

    del backend
    torch.cuda.init()
    torch.manual_seed(11)
    # The AMX INT8 MoE needs max_len (the chunk size) to be a multiple of its M_STEP.
    experts, topk, hidden, intermediate, max_rows = 8, 2, 512, 512, 64
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        gpu_experts_mask=None,
        cpuinfer_threads=8,
        threadpool_count=1,
        weight_path=weight_dir,
        chunked_prefill_size=max_rows,
        method="AMXINT8",
    )
    projections = [
        (torch.randn(experts, intermediate, hidden) * 0.05).to(torch.bfloat16),
        (torch.randn(experts, intermediate, hidden) * 0.05).to(torch.bfloat16),
        (torch.randn(experts, hidden, intermediate) * 0.05).to(torch.bfloat16),
    ]
    wrapper.load_weights_from_tensors(*projections, torch.arange(experts, dtype=torch.int64))

    def inputs(rows):
        x = (torch.randn(rows, hidden) * 0.1).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(rows)]).to(torch.int64)
        weights = torch.rand(rows, topk, dtype=torch.float32)
        return x, ids, weights

    def reference(x, ids, weights):
        output = torch.empty(x.shape[0], hidden, dtype=torch.bfloat16)
        qlen = torch.tensor([x.shape[0]], dtype=torch.int32)
        wrapper.cpu_infer.submit(
            wrapper.moe.forward_task(
                qlen.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), output.data_ptr(), False
            )
        )
        wrapper.cpu_infer.sync()
        return output

    stream = torch.cuda.Stream()

    def forward(x, ids, weights):
        with torch.cuda.stream(stream):
            wrapper.submit_forward(x, ids, weights, stream.cuda_stream)
            return wrapper.sync_forward(x, stream.cuda_stream)

    graphs = {}
    for rows in (3, 5):
        static = [t.cuda() for t in inputs(rows)]
        forward(*static)  # eager warm-up
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward(*static)
        graphs[rows] = (graph, static, output)

    mismatches = []
    for trial in range(5):
        for rows in (3, 4, 5, max_rows):
            host = inputs(rows)
            if rows in graphs:
                graph, static, output = graphs[rows]
                for dst, src in zip(static, host):
                    dst.copy_(src)
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    graph.replay()
            else:
                output = forward(*[t.cuda() for t in host])
            stream.synchronize()
            if not torch.equal(output.cpu(), reference(*host)):
                mismatches.append((trial, rows))

    x, ids, weights = [t.cuda() for t in inputs(4)]
    with torch.cuda.stream(stream):
        wrapper.submit_forward(x, ids, weights, stream.cuda_stream)
        try:
            wrapper.submit_forward(x, ids, weights, stream.cuda_stream)
            interleave_error = None
        except RuntimeError as error:
            interleave_error = str(error)
        wrapper.sync_forward(x, stream.cuda_stream)
    stream.synchronize()
    return {"mismatches": mismatches, "interleave_error": interleave_error}


def child_main():
    amx_backend.child_main(_run)


def test_graphs_and_eager_forwards_share_the_arena(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not hasattr(kt_kernel.kt_kernel_ext.moe, "AMXInt8_MOE"):
        pytest.skip("AMX INT8 extension is unavailable")
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()

    result = amx_backend.run_backend(__name__, "0", tmp_path, weight_dir=str(weight_dir))

    assert result["mismatches"] == []
    assert result["interleave_error"] is not None and "must not interleave" in result["interleave_error"]
