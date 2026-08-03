from __future__ import annotations

from typing import Mapping, Sequence


_STAGE_KEYS = (
    "pass_tb_and_synth",
    "pass_tb",
    "pass_synth",
    "pass_compile",
    "pass_parse",
)


def last_parse_valid_round_index(
    rounds: Sequence[Mapping[str, object]],
) -> int:
    """Select the newest parse-valid response without scoring later stages."""
    if not rounds:
        raise ValueError("at least one round is required")
    for index in range(len(rounds) - 1, -1, -1):
        review = rounds[index].get("review")
        if not isinstance(review, Mapping):
            continue
        code = review.get("generated_code")
        if review.get("pass_parse") is True and isinstance(code, str) and code.strip():
            return index
    return len(rounds) - 1


def best_reviewed_round_index(
    rounds: Sequence[Mapping[str, object]],
) -> int:
    """Select the earliest candidate on the best fully reviewed stage frontier."""
    if not rounds:
        raise ValueError("at least one round is required")
    best_index = 0
    best_score = tuple(False for _ in _STAGE_KEYS)
    for index, item in enumerate(rounds):
        review = item.get("review")
        if not isinstance(review, Mapping):
            continue
        code = review.get("generated_code")
        score = tuple(bool(review.get(key)) for key in _STAGE_KEYS)
        if not isinstance(code, str) or not code.strip():
            score = tuple(False for _ in _STAGE_KEYS)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index
