from __future__ import annotations

import os
import sys
from pathlib import Path


def enable_python310_hls_eval(hls_eval_root: Path) -> None:
    """Make the source checkout of HLS-Eval importable under Python 3.10.

    HLS-Eval uses the Python 3.11 ``tomllib`` module.  Bench4HLS intentionally
    remains on Python 3.10, so the standard ``tomli`` backport is registered
    under the expected module name before importing HLS-Eval.
    """

    if sys.version_info < (3, 11) and "tomllib" not in sys.modules:
        import tomli

        sys.modules["tomllib"] = tomli

    root = str(hls_eval_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    if os.environ.get("MEMENTOHLS_FILE_BACKED_DEADLINES") == "1":
        from .deadline_synth_patch import install_deadline_synth_patch

        install_deadline_synth_patch()
