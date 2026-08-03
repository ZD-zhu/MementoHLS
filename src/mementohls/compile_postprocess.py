from __future__ import annotations

import re


_ARRAY_TYPEDEF = re.compile(
    r"typedef\s+[^;]+?\b([A-Za-z_]\w*)\s*(?:\[[^\]]+\])+\s*;",
    re.DOTALL,
)


def array_typedef_names(header_text: str) -> set[str]:
    """Return typedef names whose underlying declarator is a fixed array."""
    return set(_ARRAY_TYPEDEF.findall(header_text))


def ensure_exact_header_include(
    code: str,
    header_name: str,
) -> tuple[str, list[str]]:
    exact = f'#include "{header_name}"'
    quoted = re.compile(
        rf'^\s*#\s*include\s*[<"]{re.escape(header_name)}[>"]\s*$',
        re.MULTILINE,
    )
    matches = list(quoted.finditer(code))
    if matches and matches[0].group(0).strip() == exact:
        return code, []

    actions: list[str] = []
    if matches:
        code = quoted.sub(exact, code, count=1)
        actions.append(f"normalize_project_header_include:{header_name}")
    else:
        code = exact + "\n" + code.lstrip()
        actions.append(f"add_project_header_include:{header_name}")
    return code.rstrip() + "\n", actions


def ensure_required_standard_headers(
    code: str,
    failure_log: str = "",
) -> tuple[str, list[str]]:
    """Add only the header jointly proven by code and compiler evidence."""
    requirements = (
        (
            "algorithm",
            re.compile(r"\bstd::(?:min|max)\s*\("),
            re.compile(
                r"no member named\s+[\x60\x27\"](?:min|max)[\x60\x27\"]"
                r"\s+in namespace\s+[\x60\x27\"]std[\x60\x27\"]",
                re.IGNORECASE,
            ),
        ),
        (
            "cmath",
            re.compile(r"\b(?:std::)?(?:fmin|fmax)\s*\("),
            re.compile(
                r"(?:undeclared identifier\s+[\x60\x27\"](?:fmin|fmax)[\x60\x27\"]|"
                r"no member named\s+[\x60\x27\"](?:fmin|fmax)[\x60\x27\"]"
                r"\s+in namespace\s+[\x60\x27\"]std[\x60\x27\"])",
                re.IGNORECASE,
            ),
        ),
    )
    missing = [
        header
        for header, call_pattern, diagnostic_pattern in requirements
        if call_pattern.search(code)
        and (not failure_log or diagnostic_pattern.search(failure_log))
        and not re.search(
            rf"(?m)^\s*#\s*include\s*[<\"]{re.escape(header)}[>\"]",
            code,
        )
    ]
    if not missing:
        return code, []
    prefix = "".join(f"#include <{header}>\n" for header in missing)
    return prefix + code.lstrip(), [
        f"add_standard_header:{header}" for header in missing
    ]


def remove_rejected_axis_pragmas(
    code: str,
    failure_log: str,
) -> tuple[str, list[str]]:
    """Remove only AXIS pragmas for ports named by an exact Vitis rejection."""
    diagnostic = re.compile(
        r"(?:Port\s+['\"]([A-Za-z_]\w*)['\"][^\n]*cannot be set to a AXIS|"
        r"interface mode\s+['\"]axis['\"][^\n]*function argument\s+['\"]([A-Za-z_]\w*)['\"])",
        re.IGNORECASE,
    )
    ports = sorted(
        {
            group
            for match in diagnostic.finditer(failure_log)
            for group in match.groups()
            if group
        }
    )
    repaired = code
    actions: list[str] = []
    for port in ports:
        pragma = re.compile(
            rf"(?im)^\s*#\s*pragma\s+HLS\s+INTERFACE\s+axis\b[^\n]*\bport\s*=\s*{re.escape(port)}\b[^\n]*\n?"
        )
        repaired, count = pragma.subn("", repaired)
        if count:
            actions.append(f"remove_rejected_axis_pragma:{port}:{count}")
    if not actions:
        return code, []
    return repaired.rstrip() + "\n", actions


def remove_invalid_vitis_hls_include(code: str) -> tuple[str, list[str]]:
    pattern = re.compile(
        r'^\s*#\s*include\s*[<"]vitis_hls\.h[>"]\s*\n?',
        re.MULTILINE,
    )
    repaired, count = pattern.subn("", code)
    actions = [f"remove_invalid_vitis_hls_include:{count}"] if count else []
    return repaired.rstrip() + "\n", actions


def repair_array_typedef_pointer_usage(
    code: str,
    header_text: str,
) -> tuple[str, list[str]]:
    """Repair indexing of parameters that point to fixed-array typedefs."""
    array_types = array_typedef_names(header_text)
    parameters: set[str] = set()
    for type_name in array_types:
        parameter = re.compile(
            rf"\b(?:const\s+)?{re.escape(type_name)}\s*\*\s*([A-Za-z_]\w*)"
        )
        parameters.update(parameter.findall(code))

    sites: list[tuple[str, re.Pattern[str], str]] = []
    for name in sorted(parameters, key=len, reverse=True):
        member_pattern = re.compile(
            rf"\b{re.escape(name)}\s*->\s*{re.escape(name)}\b"
        )
        index_pattern = re.compile(rf"\b{re.escape(name)}\s*\[")
        sites.extend((name, member_pattern, f"(*{name})") for _ in member_pattern.finditer(code))
        sites.extend((name, index_pattern, f"(*{name})[") for _ in index_pattern.finditer(code))
    # Every use of a parameter proven by the immutable header to be a pointer
    # to a fixed-array typedef requires the same outer dereference. Rewriting
    # all such sites is one contract-preserving operator, not a set of guessed
    # algorithm edits. Deduplicate patterns because finditer contributed one
    # entry per occurrence above.
    if not sites:
        return code, []
    repaired = code
    changed_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for name, pattern, replacement in sites:
        key = (name, pattern.pattern)
        if key in seen:
            continue
        seen.add(key)
        repaired, count = pattern.subn(replacement, repaired)
        if count:
            changed_counts[name] = changed_counts.get(name, 0) + count
    if not changed_counts:
        return code, []
    actions = [f"dereference_array_pointer:{name}:{count}" for name, count in sorted(changed_counts.items())]
    return repaired.rstrip() + "\n", actions


def _remove_balanced_function(code: str, function_name: str) -> str:
    pattern = re.compile(
        rf"(?m)^\s*(?:int|void|bool)\s+{re.escape(function_name)}\s*"
        r"\([^)]*\)\s*\{"
    )
    match = pattern.search(code)
    if not match:
        return code
    brace = code.find("{", match.start())
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(brace, len(code)):
        char = code[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return code[: match.start()] + code[index + 1 :]
    return code


def remove_illegal_main(code: str) -> tuple[str, list[str]]:
    repaired = _remove_balanced_function(code, "main")
    if repaired == code:
        return code, []
    return repaired.strip() + "\n", ["remove_illegal_main"]


def remove_host_io_calls(code: str) -> tuple[str, list[str]]:
    # Deterministic deletion is intentionally limited to standalone
    # diagnostics. Functional file/system calls require an LLM repair because
    # deleting them could silently remove algorithm inputs or state updates.
    call_line = re.compile(
        r"(?m)^\s*(?:(?:std::)?printf\s*\([^;]*\)|"
        r"(?:std::)?fprintf\s*\(\s*(?:stdout|stderr)\s*,[^;]*\))\s*;\s*$"
    )
    repaired, count = call_line.subn("", code)
    actions = [f"remove_host_io_calls:{count}"] if count else []
    return repaired.rstrip() + "\n", actions
