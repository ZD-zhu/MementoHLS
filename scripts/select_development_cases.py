#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ANCHORS = [
    "Prob050",
    "Prob087",
    "Prob068",
    "Prob038",
    "Prob012",
    "Prob105",
    "Prob060",
    "Prob094",
    "Prob076",
    "Prob102",
    "Prob018",
    "Prob009",
]
FAMILIES = [
    ("fsm_state", r"\b(?:fsm|state machine|moore|mealy|sequential)\b"),
    ("counter_shift", r"\b(?:counter|shift register|lfsr|clock|cycle)\b"),
    ("reduction_boolean", r"\b(?:reduction|popcount|truth table|boolean|parity)\b"),
    ("mux_encoder", r"\b(?:mux|multiplexer|encoder|decoder|priority)\b"),
    ("numeric_width", r"\b(?:width|signed|unsigned|shift|carry|saturat|modulo)\b"),
    ("crypto_graph", r"\b(?:encrypt|cipher|hash|graph|bfs|aes|des)\b"),
    ("signal_processing", r"\b(?:filter|fft|dct|signal|waveform)\b"),
    ("array_pointer", r"\b(?:array|pointer|in-place|in place|matrix)\b"),
    ("bit_access_pack", r"\b(?:ap_uint|bit|slice|concat|pack|unpack)\b"),
]


def family(text: str) -> str:
    for name, pattern in FAMILIES:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--pilot-count", type=int, default=34)
    args = parser.parse_args()
    prompts = args.dataset_root.resolve() / "prompts"
    entries: dict[str, dict[str, str]] = {}
    for path in sorted(prompts.glob("Prob*_prompt.txt")):
        match = re.fullmatch(r"(Prob\d{3})_prompt\.txt", path.name)
        if match is None:
            continue
        text = path.read_text(encoding="utf-8")
        entries[match.group(1)] = {
            "case": match.group(1),
            "family": family(text),
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    if sorted(entries) != [f"Prob{index:03d}" for index in range(1, 171)]:
        raise SystemExit("expected the complete public Prob001..Prob170 prompt set")
    if not set(ANCHORS).issubset(entries):
        raise SystemExit("one or more fixed smoke anchors are absent")
    if args.pilot_count < len(ANCHORS) or args.pilot_count > len(entries):
        raise SystemExit("pilot-count must be within [12, 170]")

    rng = random.Random(args.seed)
    pools: dict[str, list[str]] = defaultdict(list)
    for name, item in entries.items():
        if name not in ANCHORS:
            pools[item["family"]].append(name)
    for values in pools.values():
        rng.shuffle(values)
    selected = list(ANCHORS)
    family_order = sorted(pools)
    cursor = 0
    while len(selected) < args.pilot_count:
        name = family_order[cursor % len(family_order)]
        cursor += 1
        if pools[name]:
            selected.append(pools[name].pop())
    payload = {
        "schema_version": "dac-test-development-selection-1",
        "seed": args.seed,
        "selection_inputs": "public prompts only",
        "testbench_read": False,
        "reference_design_read": False,
        "smoke_cases": [entries[name] for name in ANCHORS],
        "pilot_cases": [entries[name] for name in selected],
        "pilot_family_counts": dict(
            sorted(Counter(entries[name]["family"] for name in selected).items())
        ),
        "pilot_seeds": [0, 1, 2, 3, 4],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
