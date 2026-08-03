#!/usr/bin/env python3
"""Build deterministic 60-rule core, 71-rule primary, and 81-rule sensitivity ERCL banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ALGORITHM_SPECIFIC = {
    "COMPILE_AP_INT_CALLABLE_BIT_ACCESS",
    "COMPILE_SIGNATURE_ARRAY_POINTER_INPLACE",
    "TB_WIDTH_SIGN_SHIFT_CARRY",
    "TB_REDUCTION_POPCOUNT_BIT_REVERSE",
    "TB_PACKED_MUX_PRIORITY_BIT_ORDER",
    "TB_STATIC_RESET_ENABLE_PRIORITY",
    "TB_OLD_NEXT_STATE_COMMIT",
    "TB_COUNTER_MODULO_CASCADE",
    "TB_SHIFT_REGISTER_LFSR_TAPS",
    "TB_FSM_TRANSITION_OUTPUT_TIMING",
    "TB_TRUTH_TABLE_WAVEFORM_BOOLEAN",
}

GENERIC_SOURCE_EXCLUDE_MARKERS = (
    "Prior v4-v7 error-class taxonomy",
    "NIST FIPS 197",
    "IEEE-754 binary64",
    "MachSuite md_grid",
)


def tier(rule_id: str) -> str:
    if rule_id.startswith("COMPILE_CONTRACT_"):
        return "exact_task_contract"
    if rule_id in ALGORITHM_SPECIFIC:
        return "algorithm_specific"
    return "generic"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-bank", type=Path, required=True)
    parser.add_argument("--generic-bank", type=Path, required=True)
    parser.add_argument("--primary-bank", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.all_bank.read_text(encoding="utf-8"))
    rules = payload["rules"]
    if len(rules) != 81:
        raise SystemExit(f"expected 81 rules, found {len(rules)}")
    if len({item["rule_id"] for item in rules}) != 81:
        raise SystemExit("rule_id values are not unique")
    for item in rules:
        item["evidence_tier"] = tier(item["rule_id"])
    counts = {name: sum(item["evidence_tier"] == name for item in rules) for name in (
        "generic", "algorithm_specific", "exact_task_contract"
    )}
    if counts != {"generic": 60, "algorithm_specific": 11, "exact_task_contract": 10}:
        raise SystemExit(f"unexpected evidence tiers: {counts}")
    payload["version"] = "dac-test-ercl-all-3.0.0"
    payload["status"] = "test-informed sensitivity bank; never used for the primary result"
    payload["evidence_tier_counts"] = counts
    payload["rules_sha256"] = canonical_sha256(rules)
    args.all_bank.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    primary = dict(payload)
    primary_rules = [
        dict(item)
        for item in rules
        if item["evidence_tier"] != "exact_task_contract"
    ]
    primary["version"] = "dac-test-ercl-primary-3.0.0"
    primary["status"] = (
        "test-informed development primary bank: task-generic and reusable "
        "algorithm-family HLS contracts; exact benchmark contracts excluded"
    )
    primary["rules"] = primary_rules
    primary["evidence_tier_counts"] = {
        "generic": 60,
        "algorithm_specific": 11,
        "exact_task_contract": 0,
    }
    primary["rules_sha256"] = canonical_sha256(primary_rules)
    args.primary_bank.write_text(
        yaml.safe_dump(primary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    generic = dict(payload)
    generic_rules = [dict(item) for item in rules if item["evidence_tier"] == "generic"]
    generic["version"] = "dac-test-ercl-core-3.0.0"
    generic["status"] = "test-informed development core ablation: task-generic HLS contracts only"
    generic["sources"] = [
        source
        for source in payload.get("sources", [])
        if not any(marker in str(source) for marker in GENERIC_SOURCE_EXCLUDE_MARKERS)
    ]
    generic["rules"] = generic_rules
    generic["evidence_tier_counts"] = {"generic": 60, "algorithm_specific": 0, "exact_task_contract": 0}
    generic["rules_sha256"] = canonical_sha256(generic_rules)
    args.generic_bank.write_text(yaml.safe_dump(generic, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"all": len(rules), "primary": len(primary_rules), "core": len(generic_rules), "tiers": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
