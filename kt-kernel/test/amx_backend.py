"""Run one AMX kernel backend per process so the A/B tests can compare them.

Kernel selection reads its `KT_*_PREFILL_AMX` flag once and caches it (see
`cached_env_flag` in operators/common.hpp): the dispatch runs once per
(expert, N-block) task, so a `getenv` per call is pure overhead. The flags are
therefore process-level, and a test that compares two backends has to give each
one its own interpreter.

Usage: the test module exposes `child_main()`, which reads its inputs from the
environment and pickles its result to `KT_AMX_OUT`; the test body calls
`run_backend()` once per backend and compares the results.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path


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
