"""merge_moe_output must write what the copy_, the add and the scale multiply wrote."""

import pytest
import torch

import kt_kernel

ext = kt_kernel.kt_kernel_ext
HIDDEN = 5120
# routed_scaling_factor of DSv4.1, GLM-5.3 / DSv3, Kimi K2.6; None is MiMo (no multiply).
SCALES = [1.5, 2.5, 2.827, None]


def _skip():
    if not torch.cuda.is_available() or not hasattr(ext, "merge_moe_output"):
        pytest.skip("kt-kernel was built without CUDA")


def _reference(gpu, host, scale):
    out = gpu + host.to(gpu.device)
    if scale is not None:
        out *= scale
    return out


def _merge(stream, gpu, host, out, scale):
    ext.merge_moe_output(
        stream.cuda_stream, gpu.data_ptr(), host.data_ptr(), out.data_ptr(), gpu.numel(),
        1.0 if scale is None else scale, scale is not None,
    )


def _assert_same_bits(actual, expected):
    a, e = actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
    mismatch = (a != e).nonzero()
    assert mismatch.numel() == 0, (
        f"{mismatch.shape[0]} elements differ; first at {mismatch[0].tolist()}: "
        f"{a[tuple(mismatch[0])].item():#06x} != {e[tuple(mismatch[0])].item():#06x}"
    )


def _all_patterns_pair(seed):
    """Every bf16 bit pattern (zeros, subnormals, infinities, NaNs) against random patterns."""
    generator = torch.Generator().manual_seed(seed)
    every = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    other = torch.randint(-32768, 32768, (every.numel(),), generator=generator, dtype=torch.int32)
    other = other.to(torch.int16).view(torch.bfloat16)
    return every.reshape(64, 1024), other.reshape(64, 1024)


@pytest.mark.parametrize("scale", SCALES)
def test_merge_matches_torch_on_every_bit_pattern(scale):
    _skip()
    stream = torch.cuda.current_stream()
    for seed in range(3):
        a, b = _all_patterns_pair(seed)
        for gpu_rows, host_rows in ((a, b), (b, a)):
            gpu = gpu_rows.cuda()
            host = gpu_rows.new_empty(host_rows.shape).pin_memory()
            host.copy_(host_rows)
            out = torch.empty_like(gpu)
            _merge(stream, gpu, host, out, scale)
            torch.cuda.synchronize()
            _assert_same_bits(out, _reference(gpu, host, scale))


@pytest.mark.parametrize("rows", [1, 3, 64])
@pytest.mark.parametrize("scale", SCALES)
def test_merge_matches_torch_on_moe_outputs(rows, scale):
    _skip()
    generator = torch.Generator().manual_seed(rows)
    gpu = (torch.randn(rows, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16).cuda()
    host = (torch.randn(rows, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16).pin_memory()
    out = torch.empty_like(gpu)
    _merge(torch.cuda.current_stream(), gpu, host, out, scale)
    torch.cuda.synchronize()
    _assert_same_bits(out, _reference(gpu, host, scale))


def test_merge_rereads_host_memory_on_graph_replay():
    """A replayed graph sees what the CPU wrote into the pinned buffer since the last replay."""
    _skip()
    rows = 2
    gpu = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    host = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16).pin_memory()
    out = torch.empty_like(gpu)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _merge(stream, gpu, host, out, 1.5)
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        gpu.copy_(torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16))
        host.copy_(torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16))
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        _assert_same_bits(out, _reference(gpu, host, 1.5))


def test_merge_reads_the_cpu_forward_of_each_replay():
    """Captured after sync_with_cuda_stream, the merge reads the output the CPU experts
    wrote in that replay, and matches the copy, add and multiply captured beside it."""
    _skip()
    if not hasattr(ext, "CPUInfer") or not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMX FP4 extension is unavailable")
    if not hasattr(ext.CPUInfer, "submit_with_cuda_stream_doorbell"):
        pytest.skip("kt-kernel was built without CUDA host callbacks")
    from test_cuda_graph_host_callback_replay import _build_moe

    experts, topk, hidden, intermediate, qlen, layers = 4, 2, 256, 256, 2, 3
    pool = ext.CPUInfer(8)
    moe, _ = _build_moe(pool, experts, topk, hidden, intermediate, 4)
    # One pinned output for every layer, as KT reuses its buffer slots.
    output = torch.zeros(qlen, hidden, dtype=torch.bfloat16).pin_memory()
    batch = torch.tensor([qlen], dtype=torch.int32).pin_memory()
    per_layer = [
        dict(
            ids=torch.zeros(qlen, topk, dtype=torch.int64).pin_memory(),
            weights=torch.zeros(qlen, topk, dtype=torch.float32).pin_memory(),
            inputs=torch.zeros(qlen, hidden, dtype=torch.bfloat16).pin_memory(),
            gpu=torch.zeros(qlen, hidden, dtype=torch.bfloat16, device="cuda"),
            merged=torch.zeros(qlen, hidden, dtype=torch.bfloat16, device="cuda"),
        )
        for _ in range(layers)
    ]

    def fill(seed):
        generator = torch.Generator().manual_seed(seed)
        for layer in per_layer:
            layer["ids"].copy_(torch.stack([torch.randperm(experts, generator=generator)[:topk] for _ in range(qlen)]))
            layer["weights"].copy_(torch.rand(qlen, topk, generator=generator))
            layer["inputs"].copy_((torch.randn(qlen, hidden, generator=generator) * 0.1).to(torch.bfloat16))
            layer["gpu"].copy_((torch.randn(qlen, hidden, generator=generator) * 0.1).to(torch.bfloat16))

    def forward(stream):
        references = []
        for layer in per_layer:
            task = moe.forward_task(
                batch.data_ptr(), topk, layer["ids"].data_ptr(), layer["weights"].data_ptr(),
                layer["inputs"].data_ptr(), output.data_ptr(), False,
            )
            pool.submit_with_cuda_stream_doorbell(stream.cuda_stream, task)
            pool.sync_with_cuda_stream(stream.cuda_stream, 0)
            _merge(stream, layer["gpu"], output, layer["merged"], 2.5)
            copied = torch.empty_like(layer["gpu"])
            copied.copy_(output, non_blocking=True)
            reference = layer["gpu"] + copied
            reference *= 2.5
            references.append(reference)
        return references

    stream = torch.cuda.Stream()
    fill(0)
    with torch.cuda.stream(stream):
        forward(stream)  # warm-up
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        references = forward(stream)
    for replay in range(10):
        fill(replay + 1)
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        for i, layer in enumerate(per_layer):
            assert references[i].abs().sum() > 0, f"replay {replay} layer {i} merged nothing"
            _assert_same_bits(layer["merged"], references[i])
