"""CUDA-graph replay of doorbell-submitted MoE forwards.

Inside a capture, `submit_with_cuda_stream_doorbell` registers the task and adds a
stream memory write that the queue worker polls; the captured sync claims an
unclaimed doorbell itself. Every replay must recompute each layer's forward on its
current inputs, in order, whether the worker or the sync callback claims it.
"""

import os
import subprocess
import sys

import pytest
import torch

import kt_kernel
from test_cuda_graph_host_callback_replay import _build_moe

ext = kt_kernel.kt_kernel_ext
LAYERS, EXPERTS, TOPK, HIDDEN, INTERMEDIATE = 3, 4, 2, 256, 256


def _replays_match_eager(qlen, replays=20):
    pool = ext.CPUInfer(8)
    moe, _ = _build_moe(pool, EXPERTS, TOPK, HIDDEN, INTERMEDIATE, max(qlen, 4))
    layers = []
    for _ in range(LAYERS):
        layers.append(
            dict(
                ids=torch.zeros(qlen, TOPK, dtype=torch.int64).pin_memory(),
                weights=torch.zeros(qlen, TOPK, dtype=torch.float32).pin_memory(),
                inputs=torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16).pin_memory(),
                output=torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16).pin_memory(),
                batch=torch.tensor([qlen], dtype=torch.int32).pin_memory(),
            )
        )

    def task(layer):
        return moe.forward_task(
            layer["batch"].data_ptr(), TOPK, layer["ids"].data_ptr(), layer["weights"].data_ptr(),
            layer["inputs"].data_ptr(), layer["output"].data_ptr(), False,
        )

    def fill(seed):
        generator = torch.Generator().manual_seed(seed)
        for layer in layers:
            layer["ids"].copy_(torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(qlen)]))
            layer["weights"].copy_(torch.rand(qlen, TOPK, generator=generator))
            layer["inputs"].copy_((torch.randn(qlen, HIDDEN, generator=generator) * 0.1).to(torch.bfloat16))
            layer["output"].zero_()

    busy = torch.zeros(1 << 20, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    fill(0)
    with torch.cuda.stream(stream):  # warm-up, eager
        for layer in layers:
            pool.submit_with_cuda_stream_doorbell(stream.cuda_stream, task(layer))
            pool.sync_with_cuda_stream(stream.cuda_stream, 0)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for layer in layers:
            pool.submit_with_cuda_stream_doorbell(stream.cuda_stream, task(layer))
            busy.mul_(1.0001)  # GPU work between submit and sync, as in decode
            pool.sync_with_cuda_stream(stream.cuda_stream, 0)
    slots = pool.doorbell_slot_count()

    for replay in range(replays):
        fill(replay + 1)
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        replayed = [layer["output"].clone() for layer in layers]
        for i, layer in enumerate(layers):
            layer["output"].zero_()
            pool.submit(task(layer))
            pool.sync()
            assert replayed[i].abs().sum() > 0, f"replay {replay} layer {i} produced no output"
            assert torch.equal(replayed[i], layer["output"]), f"replay {replay} layer {i} differs from eager"
    return slots


def _staged_replays_match_eager(qlen, replays=10):
    """Captured forwards whose staging kernel rings the doorbell (register_doorbell)."""
    pool = ext.CPUInfer(8)
    moe, _ = _build_moe(pool, EXPERTS, TOPK, HIDDEN, INTERMEDIATE, max(qlen, 4))
    layers = []
    for _ in range(LAYERS):
        layers.append(
            dict(
                hidden=torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                device_ids=torch.zeros(qlen, TOPK, dtype=torch.int32, device="cuda"),
                device_weights=torch.zeros(qlen, TOPK, dtype=torch.float32, device="cuda"),
                ids=torch.zeros(qlen, TOPK, dtype=torch.int64).pin_memory(),
                weights=torch.zeros(qlen, TOPK, dtype=torch.float32).pin_memory(),
                inputs=torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16).pin_memory(),
                output=torch.zeros(qlen, HIDDEN, dtype=torch.bfloat16).pin_memory(),
                batch=torch.tensor([qlen], dtype=torch.int32).pin_memory(),
            )
        )

    def task(layer):
        return moe.forward_task(
            layer["batch"].data_ptr(), TOPK, layer["ids"].data_ptr(), layer["weights"].data_ptr(),
            layer["inputs"].data_ptr(), layer["output"].data_ptr(), False,
        )

    def stage(stream, layer, bell, bell_value):
        ext.stage_moe_inputs(
            stream.cuda_stream, layer["hidden"].data_ptr(), layer["hidden"].numel() * 2,
            layer["device_ids"].data_ptr(), True, layer["device_weights"].data_ptr(), qlen * TOPK,
            layer["inputs"].data_ptr(), layer["ids"].data_ptr(), layer["weights"].data_ptr(), bell, bell_value,
        )

    def fill(seed):
        generator = torch.Generator().manual_seed(seed)
        for layer in layers:
            ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(qlen)])
            layer["device_ids"].copy_(ids.to(torch.int32))
            layer["device_weights"].copy_(torch.rand(qlen, TOPK, generator=generator))
            layer["hidden"].copy_((torch.randn(qlen, HIDDEN, generator=generator) * 0.1).to(torch.bfloat16))
            layer["output"].zero_()
        torch.cuda.synchronize()

    busy = torch.zeros(1 << 20, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    fill(0)
    with torch.cuda.graph(graph, stream=stream):
        for layer in layers:
            bell, bell_value = pool.register_doorbell(stream.cuda_stream, task(layer))
            assert bell != 0
            stage(stream, layer, bell, bell_value)
            busy.mul_(1.0001)  # GPU work between submit and sync, as in decode
            pool.sync_with_cuda_stream(stream.cuda_stream, 0)
    slots = pool.doorbell_slot_count()

    for replay in range(replays):
        fill(replay + 1)
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        replayed = [layer["output"].clone() for layer in layers]
        for i, layer in enumerate(layers):
            layer["output"].zero_()
            pool.submit(task(layer))
            pool.sync()
            assert replayed[i].abs().sum() > 0, f"replay {replay} layer {i} produced no output"
            assert torch.equal(replayed[i], layer["output"]), f"replay {replay} layer {i} differs from eager"
    return slots


def _skip_without_doorbell():
    if not torch.cuda.is_available() or not hasattr(ext.moe, "AMXFP4_KGroup_MOE"):
        pytest.skip("CUDA or the AMX FP4 extension is unavailable")
    if not hasattr(ext.CPUInfer, "submit_with_cuda_stream_doorbell"):
        pytest.skip("kt-kernel was built without CUDA host callbacks")


@pytest.mark.parametrize("qlen", [1, 4])
def test_doorbell_replays_match_eager(qlen):
    _skip_without_doorbell()
    assert _replays_match_eager(qlen) == LAYERS


@pytest.mark.parametrize("qlen", [1, 4])
def test_staging_kernel_doorbell_replays_match_eager(qlen):
    """The staging kernel's own doorbell store must start each replayed forward only
    once the forward's staged inputs are visible to the CPU."""
    _skip_without_doorbell()
    if not hasattr(ext, "stage_moe_inputs"):
        pytest.skip("kt-kernel was built without CUDA")
    assert _staged_replays_match_eager(qlen) == LAYERS


def _run_worker(**env):
    worker_env = dict(os.environ)
    worker_env.update(env)
    result = subprocess.run(
        [sys.executable, __file__], env=worker_env, cwd=os.path.dirname(__file__), capture_output=True, text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return int(next(line for line in result.stdout.splitlines() if line.startswith("DOORBELL_SLOTS="))[15:])


def test_doorbell_claimed_by_sync_when_worker_sleeps():
    # A worker that never spins only wakes for queued tasks, so each replay's sync
    # callback must claim the doorbell itself.
    _skip_without_doorbell()
    assert _run_worker(KT_TASKQ_SPIN_US="0") == LAYERS


def test_doorbell_disabled_falls_back_to_host_nodes():
    _skip_without_doorbell()
    assert _run_worker(KT_DOORBELL="0") == 0


if __name__ == "__main__":
    print(f"DOORBELL_SLOTS={_replays_match_eager(qlen=1, replays=10)}", flush=True)
