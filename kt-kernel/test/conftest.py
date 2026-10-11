"""Make the test helpers importable as top-level modules.

`test/` is a package, so pytest puts its parent on `sys.path` and the helpers
here would otherwise only be reachable as `test.amx_backend`. The child
processes that `amx_backend.run_backend` spawns import the test module by its
bare name, so keep this directory importable too.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
