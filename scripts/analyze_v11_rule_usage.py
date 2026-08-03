from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGES = ("pass_parse", "pass_compile", "pass_tb", "pass_synth")


def _score(review: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(review.get(stage) is True) for stage in STAGES)


def _trajectories(root: Path, modes: Iterable[str]) -> Iterable[Path]:
    for mode in modes:
        yield from sorted((root / mode).glob("**/trajectory.json"))


def analyze(root: Path, modes: list[str]) -> dict[str, Any]:
    selected = Counter()
    action_selected = Counter()
    operator_attempted = Counter()
    operator_changed = Counter()
    progress = Counter()
    success = Counter()
    regressions = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    trajectory_count = 0

    for path in _trajectories(root, modes):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = str(payload.get("mode") or path.relative_to(root).parts[0])
        rounds = list(payload.get("rounds") or [])
        trajectory_count += 1
        for index, current in enumerate(rounds[1:], start=1):
            parent = rounds[index - 1]
            selected_ids = [
                str(value) for value in current.get("selected_rule_ids") or []
            ]
            action_ids = [
                str(value) for value in current.get("action_rule_ids") or []
            ]
            selected.update(selected_ids)
            action_selected.update(action_ids)
            by_mode[mode].update(selected_ids)

            parent_score = _score(dict(parent.get("review") or {}))
            current_review = dict(current.get("review") or {})
            current_score = _score(current_review)
            for rule_id in selected_ids:
                if current_score > parent_score:
                    progress[rule_id] += 1
                elif current_score < parent_score:
                    regressions[rule_id] += 1
                if current_review.get("pass_tb_and_synth") is True:
                    success[rule_id] += 1

            operator = dict(current.get("operator") or {})
            rule_id = operator.get("authorized_rule_id")
            if rule_id and operator.get("attempted") is True:
                operator_attempted[str(rule_id)] += 1
                if operator.get("changed") is True:
                    operator_changed[str(rule_id)] += 1

    rule_ids = sorted(
        set(selected)
        | set(action_selected)
        | set(operator_attempted)
        | set(operator_changed)
    )
    rules = []
    for rule_id in rule_ids:
        selections = selected[rule_id]
        rules.append(
            {
                "rule_id": rule_id,
                "selected": selections,
                "action_selected": action_selected[rule_id],
                "operator_attempted": operator_attempted[rule_id],
                "operator_changed": operator_changed[rule_id],
                "progress": progress[rule_id],
                "success": success[rule_id],
                "regressions": regressions[rule_id],
                "progress_rate": (
                    progress[rule_id] / selections if selections else 0.0
                ),
                "success_rate": (
                    success[rule_id] / selections if selections else 0.0
                ),
            }
        )

    return {
        "root": str(root),
        "modes": modes,
        "trajectory_count": trajectory_count,
        "rules": sorted(
            rules,
            key=lambda value: (
                -int(value["selected"]),
                str(value["rule_id"]),
            ),
        ),
        "selected_by_mode": {
            mode: dict(counter.most_common()) for mode, counter in by_mode.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--modes",
        default="ercl,mementohls",
        help="Comma-separated run modes.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.root.resolve(),
        [value.strip() for value in args.modes.split(",") if value.strip()],
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
