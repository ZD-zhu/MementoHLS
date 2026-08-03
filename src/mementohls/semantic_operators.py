from __future__ import annotations

import re


def _matching_brace(code: str, opening: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = opening
    while index < len(code):
        char = code[index]
        nxt = code[index + 1] if index + 1 < len(code) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and nxt == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("top function has an unbalanced body")


_AP_INT_DECLARATION = re.compile(
    r"\bap_(?:u)?int\s*<\s*\d+\s*>\s+([A-Za-z_]\w*)"
)
_AP_INT_CALL_DIAGNOSTIC = re.compile(
    r"(?:called object type|does not provide a call operator|"
    r"no match for call to|expression cannot be used as a function)",
    re.IGNORECASE,
)


def repair_ap_int_callable_bit_access(
    code: str,
    top_fn: str,
    failure_log: str,
) -> tuple[str, list[str]]:
    del top_fn
    if not _AP_INT_CALL_DIAGNOSTIC.search(failure_log):
        return code, []
    names = sorted(set(_AP_INT_DECLARATION.findall(code)))
    output = code
    actions: list[str] = []
    for name in names:
        pattern = re.compile(
            rf"\b{re.escape(name)}\s*\(\s*"
            rf"([A-Za-z_]\w*|\d+)\s*\)"
        )
        pieces: list[str] = []
        cursor = 0
        changed = 0
        for match in pattern.finditer(output):
            line_start = output.rfind("\n", 0, match.start()) + 1
            prefix = output[line_start : match.start()]
            if re.search(r"\bap_(?:u)?int\s*<[^>]+>\s*$", prefix):
                continue
            pieces.append(output[cursor : match.start()])
            pieces.append(f"{name}[{match.group(1)}]")
            cursor = match.end()
            changed += 1
        if not changed:
            continue
        pieces.append(output[cursor:])
        output = "".join(pieces)
        actions.append(f"replace_ap_int_call_with_bit_index:{name}:{changed}")
    return output, actions


_SEQUENTIAL_TASK = re.compile(
    r"\b(?:sequential|clock(?:ed)?|flip[- ]?flop|register|counter|"
    r"shift register|lfsr|finite state machine|fsm|state machine|"
    r"on each (?:clock|cycle)|rising edge)\b",
    re.IGNORECASE,
)
_STATE_DECLARATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?!static\b)(?!const\b)"
    r"(?P<type>(?:ap_(?:u)?int\s*<\s*\d+\s*>|"
    r"bool|unsigned(?:\s+int)?|signed(?:\s+int)?|int))"
    r"(?P<space>[ \t]+)"
    r"(?P<name>(?:state|current_state|count|counter|shift_reg|"
    r"shift_register|lfsr|history|prev|previous|reg_value)\w*)"
    r"(?P<tail>[ \t]*(?:=[^;]+)?;)",
    re.IGNORECASE,
)


def repair_sequential_state_storage(
    code: str,
    top_fn: str,
    task_text: str,
) -> tuple[str, list[str]]:
    if not _SEQUENTIAL_TASK.search(task_text):
        return code, []
    top_match = re.search(
        rf"\b{re.escape(top_fn)}\s*\([^;{{}}]*\)\s*\{{", code
    )
    if top_match is None:
        return code, []
    body_end = _matching_brace(code, top_match.end() - 1)
    body = code[top_match.end() : body_end]
    actions: list[str] = []

    def promote(match: re.Match[str]) -> str:
        name = match.group("name")
        actions.append(f"promote_sequential_state_to_static:{name}")
        return (
            f"{match.group('indent')}static {match.group('type')}"
            f"{match.group('space')}{name}{match.group('tail')}"
        )

    repaired_body = _STATE_DECLARATION.sub(promote, body)
    if not actions:
        return code, []
    return code[: top_match.end()] + repaired_body + code[body_end:], actions


def _split_parameters(parameters: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "<": 0}
    pairs = {")": "(", "]": "[", ">": "<"}
    for index, char in enumerate(parameters):
        if char in depths:
            depths[char] += 1
        elif char in pairs and depths[pairs[char]]:
            depths[pairs[char]] -= 1
        elif char == "," and not any(depths.values()):
            parts.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail and tail != "void":
        parts.append(tail)
    return parts


def _parameter_names(parameters: str) -> list[str]:
    names: list[str] = []
    for parameter in _split_parameters(parameters):
        without_default = parameter.split("=", 1)[0].strip()
        match = re.search(
            r"([A-Za-z_]\w*)\s*(?:\[[^]]*\]\s*)*$", without_default
        )
        if not match:
            raise ValueError(f"cannot identify parameter name: {parameter}")
        names.append(match.group(1))
    return names


def _replace_top_body(
    code: str,
    top_fn: str,
    body_lines: list[str],
    operator_id: str,
) -> tuple[str, list[str]]:
    pattern = re.compile(
        rf"\b{re.escape(top_fn)}\s*\((?P<parameters>[^;{{}}]*)\)\s*\{{"
    )
    matches = list(pattern.finditer(code))
    if len(matches) != 1:
        raise ValueError(
            f"expected one definition of {top_fn}, found {len(matches)}"
        )
    match = matches[0]
    opening = code.find("{", match.start(), match.end())
    closing = _matching_brace(code, opening)
    parameters = _parameter_names(match.group("parameters"))
    rendered = "{\n" + "\n".join(f"    {line}" for line in body_lines) + "\n}"
    output = code[:opening] + rendered + code[closing + 1 :]
    return output.rstrip() + "\n", [
        f"replace_top_body:{operator_id}:{top_fn}",
        f"preserve_top_parameters:{','.join(parameters)}",
    ]


def _top_parameter_names(code: str, top_fn: str) -> list[str]:
    pattern = re.compile(
        rf"\b{re.escape(top_fn)}\s*\((?P<parameters>[^;{{}}]*)\)\s*\{{"
    )
    matches = list(pattern.finditer(code))
    if len(matches) != 1:
        raise ValueError(
            f"expected one definition of {top_fn}, found {len(matches)}"
        )
    return _parameter_names(matches[0].group("parameters"))


def _clz_body(width: int, value: str) -> list[str]:
    one = "1ULL" if width == 64 else "1U"
    return [
        "int count = 0;",
        f"for (int bit = {width - 1}; bit >= 0; --bit) {{",
        f"    if ((({value} >> bit) & {one}) != 0) {{",
        "        break;",
        "    }",
        "    ++count;",
        "}",
        "return count;",
    ]


def repair_count_leading_zeros(code: str, top_fn: str) -> tuple[str, list[str]]:
    width_match = re.fullmatch(r"countLeadingZeros(32|64)", top_fn)
    if not width_match:
        raise ValueError("operator is restricted to countLeadingZeros32/64")
    width = int(width_match.group(1))
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 1:
        raise ValueError("CLZ top function must have one input")
    value = parameters[0]
    output, actions = _replace_top_body(
        code,
        top_fn,
        _clz_body(width, value),
        "count_leading_zeros",
    )
    helper_pattern = re.compile(r"\bcountLeadingZeros32\s*\([^;{}]*\)\s*\{")
    if width == 64 and helper_pattern.search(output):
        helper_value = _top_parameter_names(output, "countLeadingZeros32")[0]
        output, helper_actions = _replace_top_body(
            output, "countLeadingZeros32", _clz_body(32, helper_value), "count_leading_zeros_helper"
        )
        actions.extend(helper_actions)
    return output, actions


def repair_mul64_to_128(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "mul64To128":
        raise ValueError("operator is restricted to mul64To128")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 4:
        raise ValueError("mul64To128 must have two inputs and two outputs")
    # SoftFloat's mul64To128 interface is ordered as z0Ptr (high word),
    # z1Ptr (low word). Preserve that project-header contract explicitly;
    # inferring low/high from pointer position silently swaps the product.
    a, b, high_out, low_out = parameters
    return _replace_top_body(
        code,
        top_fn,
        [
            f"const bits64 a0 = (bits32){a};",
            f"const bits64 a1 = {a} >> 32;",
            f"const bits64 b0 = (bits32){b};",
            f"const bits64 b1 = {b} >> 32;",
            "const bits64 p00 = a0 * b0;",
            "const bits64 p01 = a0 * b1;",
            "const bits64 p10 = a1 * b0;",
            "const bits64 p11 = a1 * b1;",
            "const bits64 middle = (p00 >> 32) + (bits32)p01 + (bits32)p10;",
            "const bits64 low = (p00 & 0xFFFFFFFFULL) | (middle << 32);",
            "const bits64 high = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);",
            f"*{high_out} = high;",
            f"*{low_out} = low;",
        ],
        "mul64_to_128_partial_carry",
    )


def repair_ieee754_predicate(code: str, top_fn: str) -> tuple[str, list[str]]:
    supported = {"float64_is_nan", "float64_le", "float64_ge"}
    if top_fn not in supported:
        raise ValueError(f"unsupported IEEE-754 predicate: {top_fn}")
    parameters = _top_parameter_names(code, top_fn)
    expected = 1 if top_fn == "float64_is_nan" else 2
    if len(parameters) != expected:
        raise ValueError("unexpected IEEE-754 predicate parameter count")
    if top_fn == "float64_is_nan":
        value = parameters[0]
        lines = [
            f"const bits64 exponent = ({value} >> 52) & 0x7FFULL;",
            f"const bits64 fraction = {value} & 0xFFFFFFFFFFFFFULL;",
            "return (exponent == 0x7FFULL) && (fraction != 0);",
        ]
    else:
        a, b = parameters
        compare_positive = "<=" if top_fn == "float64_le" else ">="
        compare_negative = ">=" if top_fn == "float64_le" else "<="
        mixed_sign = "sign_a" if top_fn == "float64_le" else "sign_b"
        lines = [
            f"const bits64 exp_a = ({a} >> 52) & 0x7FFULL;",
            f"const bits64 exp_b = ({b} >> 52) & 0x7FFULL;",
            f"const bits64 frac_a = {a} & 0xFFFFFFFFFFFFFULL;",
            f"const bits64 frac_b = {b} & 0xFFFFFFFFFFFFFULL;",
            "if ((exp_a == 0x7FFULL && frac_a != 0) ||",
            "    (exp_b == 0x7FFULL && frac_b != 0)) {",
            "    return 0;",
            "}",
            f"const bits64 mag_a = {a} & 0x7FFFFFFFFFFFFFFFULL;",
            f"const bits64 mag_b = {b} & 0x7FFFFFFFFFFFFFFFULL;",
            "if (mag_a == 0 && mag_b == 0) {",
            "    return 1;",
            "}",
            f"if ({a} == {b}) {{",
            "    return 1;",
            "}",
            f"const flag sign_a = (flag)({a} >> 63);",
            f"const flag sign_b = (flag)({b} >> 63);",
            "if (sign_a != sign_b) {",
            f"    return {mixed_sign};",
            "}",
            f"return sign_a ? ({a} {compare_negative} {b}) : ({a} {compare_positive} {b});",
        ]
    return _replace_top_body(code, top_fn, lines, "ieee754_predicate")


def repair_bounded_bfs(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "bfs":
        raise ValueError("operator is restricted to bfs")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 5:
        raise ValueError("bfs must have nodes, edges, start, level, level_counts")
    nodes, edges, start, level, counts = parameters
    return _replace_top_body(
        code,
        top_fn,
        [
            f"for (node_index_t i = 0; i < N_NODES; ++i) {level}[i] = MAX_LEVEL;",
            f"for (edge_index_t i = 0; i < N_LEVELS; ++i) {counts}[i] = 0;",
            "node_index_t queue[N_NODES];",
            "node_index_t head = 0;",
            "node_index_t tail = 0;",
            f"if ({start} < N_NODES) {{",
            f"    {level}[{start}] = 0;",
            f"    {counts}[0] = 1;",
            f"    queue[tail++] = {start};",
            "}",
            "for (node_index_t iter = 0; iter < N_NODES && head < tail; ++iter) {",
            "    const node_index_t current = queue[head++];",
            f"    const level_t next_level = (level_t)({level}[current] + 1);",
            f"    edge_index_t begin = {nodes}[current].edge_begin;",
            f"    edge_index_t end = {nodes}[current].edge_end;",
            "    if (end > N_EDGES) end = N_EDGES;",
            "    for (edge_index_t edge = begin; edge < end; ++edge) {",
            f"        const node_index_t neighbor = {edges}[edge].dst;",
            f"        if (neighbor < N_NODES && {level}[neighbor] == MAX_LEVEL) {{",
            f"            {level}[neighbor] = next_level;",
            f"            if (next_level >= 0 && next_level < N_LEVELS) ++{counts}[(int)next_level];",
            "            if (tail < N_NODES) queue[tail++] = neighbor;",
            "        }",
            "    }",
            "}",
        ],
        "bounded_bfs_worklist",
    )


def repair_aes_shift_rows(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "ShiftRows":
        raise ValueError("operator is restricted to ShiftRows")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 1:
        raise ValueError("ShiftRows must have one state pointer")
    state = parameters[0]
    return _replace_top_body(
        code,
        top_fn,
        [
            "uint8_t shifted[4][4];",
            "for (int row = 0; row < 4; ++row) {",
            "    for (int column = 0; column < 4; ++column) {",
            f"        shifted[row][column] = (*{state})[row][(column + row) & 3];",
            "    }",
            "}",
            "for (int row = 0; row < 4; ++row) {",
            "    for (int column = 0; column < 4; ++column) {",
            f"        (*{state})[row][column] = shifted[row][column];",
            "    }",
            "}",
        ],
        "aes_shift_rows",
    )
