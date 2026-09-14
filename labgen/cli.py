"""Command line entry point.

Not implemented. See TASKS.md T7.
"""

from __future__ import annotations

_TASK = "T7"


def _not_yet(what: str):
    raise NotImplementedError(
        f"{what} is not implemented yet -- {_TASK}. "
        f"Stages are built in the order given in TASKS.md; building this one "
        f"early means there is nothing to check its output against."
    )


def main(argv: list[str] | None = None) -> int:
    _not_yet("the labgen CLI")
