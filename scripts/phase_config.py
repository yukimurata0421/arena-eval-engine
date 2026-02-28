"""Compatibility shim for phase_config imports.

Historically modules import `phase_config` from the scripts root.
The implementation now lives in `arena.lib.phase_config`.
"""

from arena.lib.phase_config import *  # noqa: F401,F403
