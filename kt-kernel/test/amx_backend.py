"""Run one AMX kernel backend per process so the A/B tests can compare them.

Kernel selection reads its `KT_*_PREFILL_AMX` flag once and caches it (see
`cached_env_flag` in operators/common.hpp): the dispatch runs once per
(expert, N-block) task, so a `getenv` per call is pure overhead. The flags are
therefore process-level, and a test that compares two backends has to give each
one its own interpreter.

Usage: the test module exposes `child_main()`, which reads its inputs from the
environment and pickles its result to `KT_AMX_OUT`; the test body calls
`run_backend()` once per backend and compares the results.

The helpers below build the two-expert MoE these tests share, run it, and check it
against an fp32 reference.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def run_backend(module, backend, tmp_path, **params):
    """Run `module.child_main()` with `KT_*_PREFILL_AMX=backend` and return its result.

    `params` is forwarded to the child as JSON in `KT_AMX_PARAMS`.
    """
    import json

    result = Path(tmp_path) / f"{Path(module).name}-{backend}.pkl"
    env = {
        **os.environ,
        "KT_AMX_BACKEND": str(backend),
        "KT_AMX_PARAMS": json.dumps(params),
        "KT_AMX_OUT": str(result),
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).parent), *sys.path[:1], os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep),
    }
    subprocess.run(
        [sys.executable, "-c", f"import {module} as m; m.child_main()"],
        check=True,
        env=env,
        cwd=Path(__file__).parent,
    )
    with open(result, "rb") as handle:
        return pickle.load(handle)


def child_main(build_and_run):
    """Entry point for the child process: read env, run, pickle the result."""
    import json

    params = json.loads(os.environ.get("KT_AMX_PARAMS", "{}"))
    value = build_and_run(os.environ["KT_AMX_BACKEND"], **params)
    with open(os.environ["KT_AMX_OUT"], "wb") as handle:
        pickle.dump(value, handle)


def run_both(module, tmp_path, **params):
    """The (AVX512, AMX) results of `module`'s child: each backend in its own process."""
    return run_backend(module, "0", tmp_path, **params), run_backend(module, "1", tmp_path, **params)


def require_amx(moe_class):
    """Skip unless kt_kernel_ext has `moe_class` and the CPU has AMX BF16."""
    import kt_kernel

    if getattr(kt_kernel.kt_kernel_ext.moe, moe_class, None) is None:
        pytest.skip(f"{moe_class} extension is unavailable")
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        if "amx_bf16" not in cpuinfo.read():
            pytest.skip("AMX BF16 is unavailable")


def load_weights(pool, moe, experts):
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()


def build_moe(
    moe_class,
    experts,
    hidden,
    intermediate,
    qlen,
    weights,
    scales=None,
    bits=None,
    group_size=None,
    per_channel=False,
    **fields,
):
    """(pool, config, moe): a loaded `moe_class` over the (gate, up, down) `weights` on a
    16-thread pool. `fields` are further MOEConfig attributes."""
    import kt_kernel

    ext = kt_kernel.kt_kernel_ext
    pool = ext.CPUInfer(16)
    config = ext.moe.MOEConfig(experts, 1, hidden, intermediate, 0)
    config.max_len = qlen
    config.gate_proj, config.up_proj, config.down_proj = [weight.data_ptr() for weight in weights]
    if scales is None:
        config.gate_scale, config.up_scale, config.down_scale = 0, 0, 0
    else:
        config.gate_scale, config.up_scale, config.down_scale = [scale.data_ptr() for scale in scales]
    if bits is not None:
        config.quant_config.bits = bits
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
    if per_channel:
        config.quant_config.per_channel = True
    for name, value in fields.items():
        setattr(config, name, value)
    config.pool = pool.backend_
    moe = getattr(ext.moe, moe_class)(config)
    load_weights(pool, moe, experts)
    return pool, config, moe


def two_expert_routes(expert_counts):
    """Top-1 routes of expert_counts[0] tokens to expert 0 and the rest to expert 1, and
    their random weights. 33/31 exercises full and partial AMX tiles; 63/1 also the
    single-row staging path beside an aliased expert."""
    ids = torch.tensor([[0]] * expert_counts[0] + [[1]] * expert_counts[1], dtype=torch.int64).contiguous()
    weights = torch.rand(ids.shape[0], 1, dtype=torch.float32).contiguous()
    return ids, weights


def forward(pool, moe, ids, weights, inputs, rows):
    """The MoE output for the first `rows` tokens."""
    batch_size = torch.tensor([rows], dtype=torch.int32)
    output = torch.empty(inputs.shape, dtype=torch.bfloat16)
    pool.submit(
        moe.forward_task(
            batch_size.data_ptr(),
            1,
            ids.data_ptr(),
            weights.data_ptr(),
            inputs.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    pool.sync()
    return output[:rows]


def swiglu_reference(inputs, ids, weights, gate, up, down):
    """fp32 top-1 MoE over dequantized [experts, n, k] weights, rounding each GEMM and the
    activation to BF16 as the kernels do."""
    reference = torch.zeros(inputs.shape[0], down.shape[1], dtype=torch.float32)
    for expert in range(gate.shape[0]):
        selected = (ids[:, 0] == expert).nonzero(as_tuple=True)[0]
        gate_result = (inputs[selected].float() @ gate[expert].T).bfloat16().float()
        up_result = (inputs[selected].float() @ up[expert].T).bfloat16().float()
        activated = (F.silu(gate_result) * up_result).bfloat16().float()
        down_result = (activated @ down[expert].T).bfloat16().float()
        reference[selected] = weights[selected] * down_result
    return reference


def assert_matches_reference(result, reference):
    """Prefill and one-token decode within 2% mean error of the fp32 reference."""
    error = (result["prefill"].float() - reference).abs()
    assert error.mean() < reference.abs().mean() * 0.02
    assert torch.isfinite(result["decode"]).all()
    decode_error = (result["decode"].float() - reference[:1]).abs()
    assert decode_error.mean() < reference[:1].abs().mean() * 0.02
