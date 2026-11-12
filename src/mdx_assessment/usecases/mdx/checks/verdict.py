"""The verdict rule: where a flag's verdict starts, and which way it may move (E8-04).

Pure code, no LLM. Code's label (E8-01) sets the starting verdict. The student (the
Market Data Assessment Agent) may move a verdict only towards caution, that is towards
more of the analyst's attention, never away from it:

    likely_harmless  →  cant_tell  →  likely_real

| Code's label           | Starting verdict  |
| ---------------------- | ----------------- |
| `genuine_large_move`   | `likely_real`     |
| `threshold_artifact`   | `likely_harmless` |
| `ordinary`             | `likely_harmless` |
| `insufficient_history` | `cant_tell`       |

So an investigation can raise doubt about a harmless-looking flag, but it can never
talk a large move down. Code checks every draft against this before anyone sees it.
"""

from __future__ import annotations

from enum import StrEnum

from mdx_assessment.usecases.mdx.checks.near_zero import Classification


class Verdict(StrEnum):
    LIKELY_REAL = "likely_real"
    CANT_TELL = "cant_tell"
    LIKELY_HARMLESS = "likely_harmless"


CAUTION = {Verdict.LIKELY_HARMLESS: 0, Verdict.CANT_TELL: 1, Verdict.LIKELY_REAL: 2}
"""Higher asks for more of the analyst's attention."""

STARTING_VERDICT = {
    Classification.GENUINE_LARGE_MOVE: Verdict.LIKELY_REAL,
    Classification.THRESHOLD_ARTIFACT: Verdict.LIKELY_HARMLESS,
    Classification.ORDINARY: Verdict.LIKELY_HARMLESS,
    Classification.INSUFFICIENT_HISTORY: Verdict.CANT_TELL,
}


def starting_verdict(label: Classification) -> Verdict:
    return STARTING_VERDICT[label]


def lowers(start: Verdict, proposed: Verdict) -> bool:
    """True when `proposed` asks for less attention than `start`: not allowed."""
    return CAUTION[proposed] < CAUTION[start]


def at_least(verdict: Verdict, floor: Verdict) -> Verdict:
    """The more cautious of the two."""
    return verdict if CAUTION[verdict] >= CAUTION[floor] else floor
