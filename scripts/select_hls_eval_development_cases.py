#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select(dataset_root: Path, seed: int, pilot_count: int) -> dict[str, object]:
    root = dataset_root.resolve()
    by_suite: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for prompt in sorted(root.rglob("kernel_description.md")):
        case_dir = prompt.parent
        if not (case_dir / "hls_eval_config.toml").is_file():
            continue
        relative = case_dir.relative_to(root)
        if not relative.parts:
            continue
        suite = relative.parts[0]
        by_suite[suite].append((case_dir.name, prompt))
    names = [name for items in by_suite.values() for name, _ in items]
    if len(names) != 94:
        raise ValueError(f"expected 94 HLS-Eval cases, found {len(names)}")
    if len(names) != len(set(names)):
        raise ValueError("HLS-Eval development selection requires unique case names")
    if not 1 <= pilot_count <= len(names):
        raise ValueError(f"pilot_count must be in [1, {len(names)}]")

    rng = random.Random(seed)
    queues: dict[str, list[tuple[str, Path]]] = {}
    for suite, items in sorted(by_suite.items()):
        current = sorted(items)
        rng.shuffle(current)
        queues[suite] = current
    suites = sorted(queues)
    rng.shuffle(suites)
    selected: list[tuple[str, str, Path]] = []
    while len(selected) < pilot_count:
        made_progress = False
        for suite in suites:
            if queues[suite] and len(selected) < pilot_count:
                name, prompt = queues[suite].pop()
                selected.append((suite, name, prompt))
                made_progress = True
        if not made_progress:
            break
    cases = [
        {
            "case": name,
            "suite": suite,
            "prompt_sha256": _sha256(prompt),
        }
        for suite, name, prompt in selected
    ]
    return {
        "schema_version": "dac-test-hls-eval-development-selection-1",
        "seed": seed,
        "selection_inputs": "public kernel descriptions and suite paths only",
        "pilot_cases": cases,
        "pilot_count": len(cases),
        "pilot_suite_counts": dict(
            sorted(Counter(item["suite"] for item in cases).items())
        ),
        "pilot_seeds": [0, 1, 2, 3, 4],
        "testbench_read": False,
        "expected_output_read": False,
        "reference_design_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--pilot-count", type=int, default=34)
    args = parser.parse_args()
    payload = select(args.dataset_root, args.seed, args.pilot_count)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
