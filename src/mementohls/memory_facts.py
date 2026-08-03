from __future__ import annotations

import json
import re
from typing import Any


_MAX_FACT_SYMBOLS = 16
_MAX_SYNTH_ERRORS = 8
_ERROR_SYMBOLS = (
    re.compile(
        r"(?:use of undeclared identifier|unknown type name)\s+['\x60\"]?([A-Za-z_]\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\x60\"]([A-Za-z_]\w*)['\x60\"]\s+(?:was not declared|does not name a type)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:identifier|symbol)\s+['\x60\"]?([A-Za-z_]\w*)['\x60\"]?\s+is undefined",
        re.IGNORECASE,
    ),
    re.compile(
        r"no member named\s+['\x60\"]([A-Za-z_]\w*)['\x60\"]",
        re.IGNORECASE,
    ),
)
_FUNCTION_DEFINITION = (
    r"(?m)^[ \t]*(?:template\s*<[^;{}]+>\s*)?"
    r"(?:(?:static|inline|constexpr|const|volatile|signed|unsigned)\s+)*"
    r"(?:[A-Za-z_]\w*(?:\s*<[^;{}]+>)?(?:\s*[*&])?\s+)+"
    r"{name}\s*\([^;{}]*\)\s*\{"
)
_FUNCTION_CALL = r"\b{name}\s*\("
_ARRAY_TYPEDEF = re.compile(
    r"typedef\s+[^;]+?\b([A-Za-z_]\w*)\s*(?:\[[^]\n]+\])+\s*;",
    re.IGNORECASE,
)
_STRONG_SYNTH = re.compile(
    r"(?im)^.*(?:ERROR|WARNING):\s*\[[^]\n]+\].*"
    r"(?:unsupported|invalid|cannot|conflict).*$"
)
_STANDARD_LIBRARY_HEADERS = {
    "algorithm": {"min", "max"},
    "cmath": {"fmin", "fmax"},
}
_AXIS_PORT_REJECTION = re.compile(
    r"(?:Port\s+['\"]([A-Za-z_]\w*)['\"][^\n]*cannot be set to a AXIS|"
    r"interface mode\s+['\"]axis['\"][^\n]*function argument\s+['\"]([A-Za-z_]\w*)['\"])",
    re.IGNORECASE,
)
_AP_INT_DECLARATION = re.compile(
    r"\bap_(?:u)?int\s*<\s*\d+\s*>\s+([A-Za-z_]\w*)"
)
_AP_INT_CALL_DIAGNOSTIC = re.compile(
    r"(?:called object type|does not provide a call operator|"
    r"no match for call to|expression cannot be used as a function)",
    re.IGNORECASE,
)
_STATE_CANDIDATE = re.compile(
    r"(?m)^[ \t]*(?!static\b)(?:ap_(?:u)?int\s*<\s*\d+\s*>|"
    r"bool|unsigned(?:\s+int)?|signed(?:\s+int)?|int)"
    r"[ \t]+(?:state|current_state|count|counter|shift_reg|"
    r"shift_register|lfsr|history|prev|previous|reg_value)\w*"
    r"[ \t]*(?:=[^;]+)?;",
    re.IGNORECASE,
)


def _symbols(log: str) -> list[str]:
    return sorted(
        {
            match.group(1)
            for pattern in _ERROR_SYMBOLS
            for match in pattern.finditer(log)
        }
    )


def _bounded(values: list[Any], limit: int = _MAX_FACT_SYMBOLS) -> list[Any]:
    return values[:limit]


def _exact_include_present(code: str, header_name: str) -> bool:
    if not header_name:
        return False
    return (
        re.search(
            rf"(?m)^[ \t]*#[ \t]*include[ \t]*[<\"]"
            rf"{re.escape(header_name)}[>\"]",
            code,
        )
        is not None
    )


def _standard_header_for_symbol(symbol: str, code: str, log: str) -> str | None:
    """Return a missing standard header only when compiler and code agree."""
    for header_name, symbols in _STANDARD_LIBRARY_HEADERS.items():
        if symbol not in symbols:
            continue
        if header_name == "algorithm":
            call = re.search(rf"\bstd::{re.escape(symbol)}\s*\(", code)
            diagnostic = re.search(
                rf"no member named\s+['\"]{re.escape(symbol)}['\"]\s+in namespace\s+['\"]std['\"]",
                log,
                re.IGNORECASE,
            )
        else:
            call = re.search(rf"\b(?:std::)?{re.escape(symbol)}\s*\(", code)
            diagnostic = re.search(
                rf"(?:undeclared identifier\s+['\"]{re.escape(symbol)}['\"]|"
                rf"no member named\s+['\"]{re.escape(symbol)}['\"]\s+in namespace\s+['\"]std['\"])",
                log,
                re.IGNORECASE,
            )
        include_present = re.search(
            rf"(?m)^\s*#\s*include\s*[<\"]{re.escape(header_name)}[>\"]",
            code,
        )
        if call and diagnostic and not include_present:
            return header_name
    return None


def derive_memory_facts(
    *,
    stage: str,
    log: str,
    code: str,
    header: str = "",
    task: str = "",
    header_name: str = "",
) -> str:
    """Build bounded deterministic facts from allowed immutable evidence only."""
    markers: set[str] = set()
    compile_symbols: list[dict[str, Any]] = []
    compile_error_set: dict[str, Any] = {}
    exact_header_eligibility: dict[str, Any] = {
        "eligible": False,
        "header_name": header_name or None,
        "include_present": _exact_include_present(code, header_name),
        "header_owned_unresolved_symbols": [],
        "reason": "not_compile_stage",
    }
    standard_header_eligibility: dict[str, Any] = {
        "eligible": False,
        "required_headers": [],
        "symbols": [],
        "reason": "not_compile_stage",
    }
    rejected_axis_ports: list[str] = []

    if stage == "compile":
        unresolved = _symbols(log)
        helper_symbols: list[str] = []
        missing_helpers: list[str] = []
        defined_after_use: list[str] = []
        header_owned_symbols: list[str] = []
        standard_symbols: list[str] = []
        required_standard_headers: set[str] = set()

        for symbol in unresolved:
            call_positions = [
                match.start()
                for match in re.finditer(
                    _FUNCTION_CALL.format(name=re.escape(symbol)), code
                )
            ]
            definitions = [
                match.start()
                for match in re.finditer(
                    _FUNCTION_DEFINITION.replace("{name}", re.escape(symbol)),
                    code,
                )
            ]
            header_owned = (
                re.search(rf"\b{re.escape(symbol)}\b", header) is not None
            )
            standard_header = _standard_header_for_symbol(symbol, code, log)
            if standard_header:
                status = "standard_library_header_missing"
                standard_symbols.append(symbol)
                required_standard_headers.add(standard_header)
                markers.add("compile_standard_library_header_missing")
            elif header_owned:
                status = "header_owned"
                header_owned_symbols.append(symbol)
            elif call_positions and definitions and min(definitions) > min(
                call_positions
            ):
                status = "defined_after_use"
                helper_symbols.append(symbol)
                defined_after_use.append(symbol)
                markers.add("compile_helper_defined_after_use")
            elif call_positions and not definitions:
                status = "missing_definition"
                helper_symbols.append(symbol)
                missing_helpers.append(symbol)
                markers.add("compile_missing_helper_definition")
            elif call_positions:
                status = "definition_visible"
            else:
                status = "unclassified"
            compile_symbols.append(
                {
                    "symbol": symbol,
                    "status": status,
                    "is_function_call": bool(call_positions),
                }
            )

        if unresolved:
            markers.add("compile_unresolved_symbol_set")
        if len(unresolved) >= 2:
            markers.add("compile_multiple_unresolved_symbols")
        if helper_symbols:
            markers.add("compile_unresolved_helper")
        if len(helper_symbols) >= 2:
            markers.add("compile_unresolved_helper_closure")

        helper_closure = {
            "required": len(helper_symbols) >= 2,
            "helper_count": len(helper_symbols),
            "helper_symbols": _bounded(helper_symbols),
            "missing_definition_symbols": _bounded(missing_helpers),
            "defined_after_use_symbols": _bounded(defined_after_use),
            "repair_contract": (
                "resolve_all_reported_helpers_in_one_bounded_candidate"
                if len(helper_symbols) >= 2
                else "resolve_reported_helper"
            ),
        }
        compile_error_set = {
            "unresolved_symbol_count": len(unresolved),
            "unresolved_symbols": _bounded(unresolved),
            "header_owned_symbols": _bounded(header_owned_symbols),
            "helper_closure": helper_closure,
            "omitted_symbol_count": max(0, len(unresolved) - _MAX_FACT_SYMBOLS),
        }

        include_present = _exact_include_present(code, header_name)
        eligible = bool(header_name and header_owned_symbols and not include_present)
        if eligible:
            reason = "header_owned_unresolved_symbol_and_exact_include_absent"
            markers.add("compile_exact_header_include_eligible")
        elif not header_name:
            reason = "header_name_unavailable"
        elif include_present:
            reason = "exact_include_already_present"
        elif not header_owned_symbols:
            reason = "no_header_owned_unresolved_symbol"
        else:
            reason = "ineligible"
        exact_header_eligibility = {
            "eligible": eligible,
            "header_name": header_name or None,
            "include_present": include_present,
            "header_owned_unresolved_symbols": _bounded(header_owned_symbols),
            "reason": reason,
        }

        standard_header_eligibility = {
            "eligible": bool(required_standard_headers),
            "required_headers": sorted(required_standard_headers),
            "symbols": _bounded(standard_symbols),
            "reason": (
                "compiler_and_code_prove_missing_standard_header"
                if required_standard_headers
                else "no_supported_standard_header_diagnostic"
            ),
        }

        array_types = _ARRAY_TYPEDEF.findall(header)
        for type_name in array_types:
            pointer_params = re.findall(
                rf"\b{re.escape(type_name)}\s*\*\s*([A-Za-z_]\w*)", header
            )
            for parameter in pointer_params:
                wrong_index = re.search(
                    rf"(?<!\*)\b{re.escape(parameter)}\s*\[", code
                )
                wrong_member = re.search(
                    rf"\b{re.escape(parameter)}\s*->", code
                )
                if wrong_index or wrong_member:
                    markers.add("fixed_array_typedef_pointer_misuse")
        if _AP_INT_CALL_DIAGNOSTIC.search(log):
            declared_ap_ints = sorted(set(_AP_INT_DECLARATION.findall(code)))
            callable_sites = [
                name
                for name in declared_ap_ints
                if re.search(
                    rf"\b{re.escape(name)}\s*\(\s*"
                    rf"(?:[A-Za-z_]\w*|\d+)\s*\)",
                    code,
                )
            ]
            if callable_sites:
                markers.add("compile_ap_int_callable_bit_access")

    task_capsules = {
        "ieee754_bit_exact": (
            r"IEEE\s*754|signaling\s+NaN|quiet\s+NaN|exponent\s+field|"
            r"fraction\s+field|mantissa"
        ),
        "in_place_fixed_array": (
            r"in[- ]place|modif(?:y|ies) the input|updates? the .*array"
        ),
        "multi_axis_mapping": r"row|column|channel|flatten|\b[234]D\s+array",
        "reduction_contract": (
            r"reduction|accumulat|dot product|sum of|maximum|minimum"
        ),
        "bit_access_pack": (
            r"\b(?:bit|slice|range|pack|unpack|concatenat|"
            r"population count|popcount|reverse)\b"
        ),
        "sequential_state": (
            r"\b(?:sequential|clock(?:ed)?|flip[- ]?flop|register|"
            r"on each (?:clock|cycle)|rising edge)\b"
        ),
        "reset_enable_priority": (
            r"\b(?:synchronous reset|asynchronous reset|reset|enable)\b"
        ),
        "counter_contract": (
            r"\b(?:counter|count from|modulo|decade|saturat|cascade)\b"
        ),
        "shift_lfsr": (
            r"\b(?:shift register|lfsr|linear feedback|tap|feedback bit)\b"
        ),
        "fsm_contract": (
            r"\b(?:finite state machine|fsm|moore|mealy|state transition)\b"
        ),
        "mux_encoder": (
            r"\b(?:multiplexer|mux|priority encoder|encoder|decoder|select)\b"
        ),
        "truth_waveform": (
            r"\b(?:truth table|waveform|timing diagram|boolean function)\b"
        ),
        "width_carry": (
            r"\b(?:carry|overflow|signed|unsigned|bit width|"
            r"\d+[- ]bit add|\d+[- ]bit subtract)\b"
        ),
        "signature_array_inplace": (
            r"\b(?:function signature|prototype|array|pointer|"
            r"in[- ]place|TopModule)\b"
        ),
    }
    capsules = sorted(
        name
        for name, pattern in task_capsules.items()
        if re.search(pattern, task, re.IGNORECASE)
    )
    markers.update(f"task_{name}" for name in capsules)
    if (
        "sequential_state" in capsules
        and _STATE_CANDIDATE.search(code)
        and not re.search(
            r"(?m)^[ \t]*static[ \t]+(?:ap_(?:u)?int|bool|unsigned|"
            r"signed|int)",
            code,
        )
    ):
        markers.add("code_nonstatic_sequential_state_candidate")

    synth_errors: list[str] = []
    if stage == "synth":
        synth_errors = [
            " ".join(match.group(0).split())[:360]
            for match in _STRONG_SYNTH.finditer(log)
        ][:_MAX_SYNTH_ERRORS]
        if any(
            re.search(
                r"pointer|interface|argument|port|top function",
                line,
                re.IGNORECASE,
            )
            for line in synth_errors
        ):
            markers.add("synth_exact_pointer_or_interface_error")
        if any(
            re.search(r"pragma|directive", line, re.IGNORECASE)
            for line in synth_errors
        ):
            markers.add("synth_exact_pragma_error")
        rejected_axis_ports = sorted(
            {
                group
                for match in _AXIS_PORT_REJECTION.finditer(log)
                for group in match.groups()
                if group
            }
        )
        if rejected_axis_ports:
            markers.add("synth_exact_axis_port_rejection")

    payload = {
        "schema_version": "2",
        "stage": stage,
        "markers": sorted(markers),
        "compile_symbols": _bounded(compile_symbols),
        "compile_error_set": compile_error_set,
        "eligibility": {
            "exact_header_include": exact_header_eligibility,
            "standard_library_header": standard_header_eligibility,
            "prefilter_policy": "eligibility_before_ranking_and_top_k",
        },
        "task_capsules": capsules,
        "synth_error_lines": synth_errors,
        "rejected_axis_ports": rejected_axis_ports,
        "reference_kernel_read_or_used": False,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)
