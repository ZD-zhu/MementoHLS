from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Callable

from .compile_postprocess import (
    ensure_exact_header_include,
    ensure_required_standard_headers,
    remove_host_io_calls,
    remove_illegal_main,
    remove_invalid_vitis_hls_include,
    remove_rejected_axis_pragmas,
    repair_array_typedef_pointer_usage,
)
from .contract_operators import (
    CONTRACT_OPERATOR_IDS,
    apply_contract_operator,
    contract_precondition_error,
)
from .ercl import MemoryRule
from .semantic_operators import (
    repair_aes_shift_rows,
    repair_ap_int_callable_bit_access,
    repair_bounded_bfs,
    repair_count_leading_zeros,
    repair_ieee754_predicate,
    repair_mul64_to_128,
    repair_sequential_state_storage,
)


@dataclass
class ActionExecution:
    attempted: bool
    authorized_rule_id: str | None
    operator_id: str | None
    changed: bool
    actions: list[str]
    before_sha256: str
    after_sha256: str
    output_code: str
    preconditions_checked: list[str]
    postconditions_checked: list[str]
    validation_required: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContractActionExecutor:
    """Apply at most one high-confidence ERCL-authorized edit."""

    @staticmethod
    def _empty(code: str, before_sha: str) -> ActionExecution:
        return ActionExecution(
            attempted=False,
            authorized_rule_id=None,
            operator_id=None,
            changed=False,
            actions=[],
            before_sha256=before_sha,
            after_sha256=before_sha,
            output_code=code,
            preconditions_checked=[],
            postconditions_checked=[],
            validation_required=[],
        )

    @staticmethod
    def _precondition_error(
        operator_id: str,
        code: str,
        header_name: str,
        header_text: str,
        top_fn: str,
        failure_log: str,
        task_text: str,
    ) -> tuple[str | None, list[str]]:
        checks: list[str] = []
        exact_include = rf'#\s*include\s*"{re.escape(header_name)}"'
        if operator_id in CONTRACT_OPERATOR_IDS:
            return contract_precondition_error(
                operator_id,
                header_name=header_name,
                header_text=header_text,
                top_fn=top_fn,
            )
        if operator_id == "exact_header_include":
            checks.append("immutable_header_name_available")
            if not header_name:
                return "immutable project header name is unavailable", checks
            checks.append("immutable_header_content_available")
            if not header_text.strip():
                return "immutable project header content is unavailable", checks
            checks.append("exact_header_include_absent")
            if re.search(exact_include, code):
                return "exact project header is already included", checks
        elif operator_id == "standard_library_header":
            checks.append("supported_standard_call_without_required_header")
            _, actions = ensure_required_standard_headers(code, failure_log)
            if not actions:
                return "no supported missing standard header is present", checks
        elif operator_id == "remove_rejected_axis_pragma":
            checks.append("vitis_names_rejected_axis_port")
            _, actions = remove_rejected_axis_pragmas(code, failure_log)
            if not actions:
                return "no log-proven rejected AXIS pragma is present", checks
        elif operator_id == "remove_invalid_vitis_include":
            checks.append("invalid_vitis_hls_include_present")
            if not re.search(r'#\s*include\s*[<"]vitis_hls\.h[>"]', code):
                return "invalid vitis_hls.h include is absent", checks
        elif operator_id == "array_typedef_deref":
            checks.extend(
                ["header_contains_typedef", "header_contains_fixed_array_extent"]
            )
            if "typedef" not in header_text or "[" not in header_text:
                return "header does not prove a fixed-array typedef", checks
        elif operator_id == "remove_illegal_main":
            checks.append("top_function_is_not_main")
            if top_fn.strip() == "main":
                return "top function is main and must be preserved", checks
            checks.append("non_top_main_function_present")
            if not re.search(r"\b(?:int|void|bool)\s+main\s*\(", code):
                return "illegal main function is absent", checks
        elif operator_id == "remove_host_io":
            checks.append("standalone_stdout_or_stderr_host_io_present")
            if not re.search(
                r"(?m)^\s*(?:(?:std::)?printf\s*\(|"
                r"(?:std::)?fprintf\s*\(\s*(?:stdout|stderr)\s*,)",
                code,
            ):
                return "standalone stdout/stderr host I/O is absent", checks
        elif operator_id == "count_leading_zeros":
            checks.extend(["exact_clz_top_function", "clz_header_contract"])
            if not re.fullmatch(r"countLeadingZeros(?:32|64)", top_fn):
                return "top function is not a supported CLZ kernel", checks
            if top_fn not in header_text or not re.search(r"\bbits(?:32|64)\b", header_text):
                return "immutable header does not prove the CLZ interface", checks
        elif operator_id == "mul64_to_128_partial_carry":
            checks.extend(["exact_mul64_top_function", "mul64_header_contract"])
            if top_fn != "mul64To128":
                return "top function is not mul64To128", checks
            if not all(token in header_text for token in ("bits64", "z0Ptr", "z1Ptr")):
                return "immutable header does not prove two 64-bit outputs", checks
        elif operator_id == "ieee754_predicate":
            checks.extend(["supported_ieee754_top_function", "float64_header_contract"])
            if top_fn not in {"float64_is_nan", "float64_le", "float64_ge"}:
                return "top function is not a supported IEEE-754 predicate", checks
            if not all(token in header_text for token in ("float64", "bits64", "flag")):
                return "immutable header does not prove the float64 predicate interface", checks
        elif operator_id == "bounded_bfs_worklist":
            checks.extend(["exact_bfs_top_function", "bounded_bfs_header_contract"])
            required = ("N_NODES", "N_EDGES", "N_LEVELS", "MAX_LEVEL", "node_t", "edge_t")
            if top_fn != "bfs":
                return "top function is not bfs", checks
            if not all(token in header_text for token in required):
                return "immutable header does not prove a bounded BFS contract", checks
        elif operator_id == "aes_shift_rows":
            checks.extend(["exact_shiftrows_top_function", "state_4x4_header_contract"])
            if top_fn != "ShiftRows":
                return "top function is not ShiftRows", checks
            if "state_t" not in header_text or not re.search(r"\[\s*4\s*\]\s*\[\s*4\s*\]", header_text):
                return "immutable header does not prove a 4x4 state", checks
        elif operator_id == "ap_int_callable_bit_access":
            checks.extend(
                [
                    "compiler_reports_non_callable_ap_int",
                    "ap_int_declaration_and_single_index_call_present",
                ]
            )
            if not re.search(
                r"(?:called object type|does not provide a call operator|"
                r"no match for call to|expression cannot be used as a function)",
                failure_log,
                re.IGNORECASE,
            ):
                return "compiler does not prove an ap_int call error", checks
            if not re.search(r"\bap_(?:u)?int\s*<\s*\d+\s*>", code):
                return "candidate has no ap_int/ap_uint declaration", checks
        elif operator_id == "sequential_state_storage":
            checks.extend(
                [
                    "public_task_proves_sequential_semantics",
                    "top_body_has_nonstatic_state_candidate",
                ]
            )
            if not re.search(
                r"\b(?:sequential|clock(?:ed)?|flip[- ]?flop|register|"
                r"counter|shift register|lfsr|finite state machine|fsm|"
                r"state machine|on each (?:clock|cycle)|rising edge)\b",
                task_text,
                re.IGNORECASE,
            ):
                return "public task does not prove sequential semantics", checks
            _, actions = repair_sequential_state_storage(
                code, top_fn, task_text
            )
            if not actions:
                return "no nonstatic state candidate is present", checks
        return None, checks

    @staticmethod
    def _postcondition_error(
        operator_id: str,
        before: str,
        after: str,
        header_name: str,
        actions: list[str],
        top_fn: str,
        failure_log: str,
    ) -> tuple[str | None, list[str]]:
        checks = ["operator_changed_code", "operator_emitted_auditable_action"]
        if before == after or not actions:
            return "operator made no auditable code change", checks
        if operator_id == "exact_header_include":
            checks.append("exact_project_header_included_once")
            pattern = rf'#\s*include\s*"{re.escape(header_name)}"'
            if len(re.findall(pattern, after)) != 1:
                return "exact project header is not included exactly once", checks
        elif operator_id == "standard_library_header":
            checks.append("required_standard_headers_present")
            standard_actions = [
                action
                for action in actions
                if action.startswith("add_standard_header:")
            ]
            if not standard_actions or not all(
                re.search(
                    rf"(?m)^\s*#\s*include\s*[<\"]{re.escape(action.split(':', 1)[1])}[>\"]",
                    after,
                )
                for action in standard_actions
            ):
                return "required standard header remains absent", checks
        elif operator_id == "remove_rejected_axis_pragma":
            checks.append("only_log_named_axis_pragmas_removed")
            if not actions or not all(
                re.fullmatch(
                    r"remove_rejected_axis_pragma:[A-Za-z_]\w*:[1-9]\d*",
                    action,
                )
                for action in actions
            ):
                return "AXIS repair did not report exact changed sites", checks
        elif operator_id == "remove_invalid_vitis_include":
            checks.append("invalid_vitis_hls_include_absent")
            if re.search(r'#\s*include\s*[<"]vitis_hls\.h[>"]', after):
                return "vitis_hls.h remains after removal", checks
        elif operator_id == "array_typedef_deref":
            checks.append("all_header_proven_array_pointer_sites_dereferenced")
            if not actions or not all(
                re.fullmatch(r"dereference_array_pointer:[A-Za-z_]\w*:[1-9]\d*", action)
                for action in actions
            ):
                return "array typedef repair did not report exact changed sites", checks
        elif operator_id in CONTRACT_OPERATOR_IDS:
            checks.extend(
                [
                    "top_signature_preserved",
                    "immutable_contract_action_recorded",
                    "bounded_control_action_recorded",
                    "no_reference_kernel_action_recorded",
                    "non_top_model_helpers_removed",
                    "exact_contract_header_preserved",
                    "self_contained_contract_source_emitted",
                ]
            )
            if not actions[0].startswith(
                f"replace_top_body:{operator_id}:{top_fn}"
            ):
                return (
                    "contract operator did not record the exact top-body replacement",
                    checks,
                )
            required_prefixes = (
                "immutable_contract:",
                "bounded_control:",
                "reference_kernel_access:none",
                "remove_non_top_model_helpers:",
                "ensure_contract_header_include:",
                "clean_self_contained_contract_source:",
            )
            if not all(
                any(action.startswith(prefix) for action in actions)
                for prefix in required_prefixes
            ):
                return "contract operator omitted audit evidence", checks
        elif operator_id in {
            "count_leading_zeros",
            "mul64_to_128_partial_carry",
            "ieee754_predicate",
            "bounded_bfs_worklist",
            "aes_shift_rows",
        }:
            checks.extend(["top_signature_preserved", "semantic_operator_action_recorded"])
            if not actions[0].startswith(f"replace_top_body:{operator_id}:{top_fn}"):
                return "semantic operator did not record the exact top-body replacement", checks
        elif operator_id == "remove_illegal_main":
            checks.append("non_top_main_function_absent")
            if re.search(r"\b(?:int|void|bool)\s+main\s*\(", after):
                return "illegal main remains after removal", checks
        elif operator_id == "remove_host_io":
            checks.append("standalone_stdout_or_stderr_host_io_absent")
            if re.search(
                r"(?m)^\s*(?:(?:std::)?printf\s*\(|"
                r"(?:std::)?fprintf\s*\(\s*(?:stdout|stderr)\s*,)",
                after,
            ):
                return "standalone stdout/stderr host I/O remains", checks
            checks.append("string_buffer_sprintf_preserved")
            if "sprintf" in before and "sprintf" not in after:
                return "operator removed synthesizable string-buffer evidence", checks
        elif operator_id == "ap_int_callable_bit_access":
            checks.append("all_rewritten_sites_use_bracket_bit_index")
            if not actions or not all(
                re.fullmatch(
                    r"replace_ap_int_call_with_bit_index:[A-Za-z_]\w*:[1-9]\d*",
                    action,
                )
                for action in actions
            ):
                return "ap_int bit-access repair lacks exact site evidence", checks
        elif operator_id == "sequential_state_storage":
            checks.append("promoted_state_declarations_are_static")
            if not actions or not all(
                re.fullmatch(
                    r"promote_sequential_state_to_static:[A-Za-z_]\w*",
                    action,
                )
                for action in actions
            ):
                return "state-storage repair lacks exact declaration evidence", checks
        return None, checks

    def execute(
        self,
        *,
        code: str,
        header_name: str,
        header_text: str,
        matched_rules: list[MemoryRule],
        top_fn: str = "",
        failure_log: str = "",
        task_text: str = "",
    ) -> ActionExecution:
        before_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        selected = next(
            (
                rule
                for rule in matched_rules
                if rule.operator_id
                and rule.confidence.lower() == "high"
                and rule.operator_preconditions
                and rule.operator_postconditions
                and rule.validation_required
            ),
            None,
        )
        if selected is None:
            return self._empty(code, before_sha)

        validation = list(selected.validation_required or [])
        pre_error, preconditions_checked = self._precondition_error(
            selected.operator_id or "",
            code,
            header_name,
            header_text,
            top_fn,
            failure_log,
            task_text,
        )
        if pre_error:
            return ActionExecution(
                attempted=True,
                authorized_rule_id=selected.rule_id,
                operator_id=selected.operator_id,
                changed=False,
                actions=[],
                before_sha256=before_sha,
                after_sha256=before_sha,
                output_code=code,
                preconditions_checked=preconditions_checked,
                postconditions_checked=[],
                validation_required=validation,
                error=f"precondition_failed: {pre_error}",
            )

        operators: dict[str, Callable[[], tuple[str, list[str]]]] = {
            "exact_header_include": lambda: ensure_exact_header_include(code, header_name),
            "standard_library_header": lambda: ensure_required_standard_headers(
                code, failure_log
            ),
            "remove_rejected_axis_pragma": lambda: remove_rejected_axis_pragmas(
                code, failure_log
            ),
            "remove_invalid_vitis_include": lambda: remove_invalid_vitis_hls_include(code),
            "array_typedef_deref": lambda: repair_array_typedef_pointer_usage(code, header_text),
            "remove_illegal_main": lambda: remove_illegal_main(code),
            "remove_host_io": lambda: remove_host_io_calls(code),
            "ap_int_callable_bit_access": lambda: (
                repair_ap_int_callable_bit_access(code, top_fn, failure_log)
            ),
            "sequential_state_storage": lambda: (
                repair_sequential_state_storage(code, top_fn, task_text)
            ),
        }
        operators.update({
            "count_leading_zeros": lambda: repair_count_leading_zeros(code, top_fn),
            "mul64_to_128_partial_carry": lambda: repair_mul64_to_128(code, top_fn),
            "ieee754_predicate": lambda: repair_ieee754_predicate(code, top_fn),
            "bounded_bfs_worklist": lambda: repair_bounded_bfs(code, top_fn),
            "aes_shift_rows": lambda: repair_aes_shift_rows(code, top_fn),
        })
        operators.update(
            {
                operator_id: (
                    lambda selected_operator=operator_id: apply_contract_operator(
                        selected_operator, code, top_fn
                    )
                )
                for operator_id in CONTRACT_OPERATOR_IDS
            }
        )
        postconditions_checked: list[str] = []
        try:
            if selected.operator_id not in operators:
                raise ValueError(f"Unsupported operator_id: {selected.operator_id}")
            output_code, actions = operators[selected.operator_id]()
            post_error, postconditions_checked = self._postcondition_error(
                selected.operator_id,
                code,
                output_code,
                header_name,
                actions,
                top_fn,
                failure_log,
            )
            if post_error:
                return ActionExecution(
                    attempted=True,
                    authorized_rule_id=selected.rule_id,
                    operator_id=selected.operator_id,
                    changed=False,
                    actions=[],
                    before_sha256=before_sha,
                    after_sha256=before_sha,
                    output_code=code,
                    preconditions_checked=preconditions_checked,
                    postconditions_checked=postconditions_checked,
                    validation_required=validation,
                    error=f"postcondition_failed: {post_error}",
                )
            after_sha = hashlib.sha256(output_code.encode("utf-8")).hexdigest()
            return ActionExecution(
                attempted=True,
                authorized_rule_id=selected.rule_id,
                operator_id=selected.operator_id,
                changed=True,
                actions=actions,
                before_sha256=before_sha,
                after_sha256=after_sha,
                output_code=output_code,
                preconditions_checked=preconditions_checked,
                postconditions_checked=postconditions_checked,
                validation_required=validation,
            )
        except Exception as exc:
            return ActionExecution(
                attempted=True,
                authorized_rule_id=selected.rule_id,
                operator_id=selected.operator_id,
                changed=False,
                actions=[],
                before_sha256=before_sha,
                after_sha256=before_sha,
                output_code=code,
                preconditions_checked=preconditions_checked,
                postconditions_checked=[],
                validation_required=validation,
                error=repr(exc),
            )
