"""The gates. Every check runs without an LLM in the loop.

Not implemented. See TASKS.md T3.
"""

from __future__ import annotations

_TASK = "T3"


def _not_yet(what: str):
    raise NotImplementedError(
        f"{what} is not implemented yet -- {_TASK}. "
        f"Stages are built in the order given in TASKS.md; building this one "
        f"early means there is nothing to check its output against."
    )
