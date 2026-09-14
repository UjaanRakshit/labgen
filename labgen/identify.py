"""Video -> [ObjectObservation] via a VLM.

Not implemented. See TASKS.md T6.
"""

from __future__ import annotations

_TASK = "T6"


def _not_yet(what: str):
    raise NotImplementedError(
        f"{what} is not implemented yet -- {_TASK}. "
        f"Stages are built in the order given in TASKS.md; building this one "
        f"early means there is nothing to check its output against."
    )
