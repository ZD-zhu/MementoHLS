from __future__ import annotations

import re
from typing import Callable

from .semantic_operators import (
    _matching_brace,
    _replace_top_body,
    _top_parameter_names,
    repair_ieee754_predicate,
)


CONTRACT_CORDIC_FIXED_SHIFT = "contract_cordic_fixed_shift"
CONTRACT_FLOAT64_NATIVE_MULTIPLY = "contract_float64_native_multiply"
CONTRACT_NEEDWUN_ROW_MAJOR_DP = "contract_needwun_row_major_dp"
CONTRACT_PRESENT80_STANDARD = "contract_present80_standard"
CONTRACT_DES_FEISTEL_SCHEDULE = "contract_des_feistel_schedule"
CONTRACT_FLOAT64_NATIVE_DIVIDE = "contract_float64_native_divide"
CONTRACT_AES128_CIPHER = "contract_aes128_cipher"
CONTRACT_MD_GRID_BOUNDED_LJ = "contract_md_grid_bounded_lj"
CONTRACT_FLOAT64_GE_PREDICATE = "contract_float64_ge_predicate"
CONTRACT_FLOAT64_LE_PREDICATE = "contract_float64_le_predicate"

CONTRACT_OPERATOR_IDS = frozenset(
    {
        CONTRACT_CORDIC_FIXED_SHIFT,
        CONTRACT_FLOAT64_NATIVE_MULTIPLY,
        CONTRACT_NEEDWUN_ROW_MAJOR_DP,
        CONTRACT_PRESENT80_STANDARD,
        CONTRACT_DES_FEISTEL_SCHEDULE,
        CONTRACT_FLOAT64_NATIVE_DIVIDE,
        CONTRACT_AES128_CIPHER,
        CONTRACT_MD_GRID_BOUNDED_LJ,
        CONTRACT_FLOAT64_GE_PREDICATE,
        CONTRACT_FLOAT64_LE_PREDICATE,
    }
)


_CONTRACT_HEADERS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    CONTRACT_CORDIC_FIXED_SHIFT: (
        "cordic",
        "cordic.h",
        (
            r"typedef\s+ap_fixed\s*<\s*12\s*,\s*2\s*>\s+THETA_TYPE\s*;",
            r"typedef\s+ap_fixed\s*<\s*12\s*,\s*2\s*>\s+COS_SIN_TYPE\s*;",
            r"const\s+int\s+NUM_ITERATIONS\s*=\s*32\s*;",
            r"static\s+THETA_TYPE\s+cordic_phase\s*\[\s*64\s*\]",
            r"void\s+cordic\s*\(\s*THETA_TYPE\s+\w+\s*,\s*COS_SIN_TYPE\s*&\s*\w+\s*,\s*COS_SIN_TYPE\s*&\s*\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_FLOAT64_NATIVE_MULTIPLY: (
        "float64_mul",
        "dfmul.h",
        (
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+bits64\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+float64\s*;",
            r"float64\s+float64_mul\s*\(\s*float64\s+\w+\s*,\s*float64\s+\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_NEEDWUN_ROW_MAJOR_DP: (
        "needwun",
        "nw_nw.h",
        (
            r"#\s*define\s+ALEN\s+128\b",
            r"#\s*define\s+BLEN\s+128\b",
            r"void\s+needwun\s*\(",
            r"int\s+\w+\s*\[\s*\(\s*ALEN\s*\+\s*1\s*\)\s*\*\s*\(\s*BLEN\s*\+\s*1\s*\)\s*\]",
            r"char\s+\w+\s*\[\s*\(\s*ALEN\s*\+\s*1\s*\)\s*\*\s*\(\s*BLEN\s*\+\s*1\s*\)\s*\]",
        ),
    ),
    CONTRACT_PRESENT80_STANDARD: (
        "present80_encryptBlock",
        "present.h",
        (
            r"#\s*define\s+PRESENT_80_KEY_SIZE_BYTES\s+10\b",
            r"#\s*define\s+PRESENT_BLOCK_SIZE_BYTES\s+8\b",
            r"#\s*define\s+ROUNDS\s+32\b",
            r"typedef\s+unsigned\s+char\s+present_key_t\s*\[\s*PRESENT_80_KEY_SIZE_BYTES\s*\]\s*;",
            r"typedef\s+unsigned\s+char\s+block_t\s*\[\s*PRESENT_BLOCK_SIZE_BYTES\s*\]\s*;",
            r"static\s+const\s+unsigned\s+char\s+sBox\s*\[\s*16\s*\]\s*=",
            r"void\s+present80_encryptBlock\s*\(\s*block_t\s*\*\s*\w+\s*,\s*present_key_t\s*\*\s*\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_DES_FEISTEL_SCHEDULE: (
        "des_crypt",
        "des.h",
        (
            r"typedef\s+unsigned\s+char\s+des_key_t\s*\[\s*16\s*\]\s*\[\s*6\s*\]\s*;",
            r"typedef\s+unsigned\s+char\s+des_block_t\s*\[\s*8\s*\]\s*;",
            r"typedef\s+unsigned\s+int\s+des_state_t\s*\[\s*2\s*\]\s*;",
            r"#\s*define\s+SBOXBIT\s*\(",
            r"const\s+unsigned\s+char\s+sbox1\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox2\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox3\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox4\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox5\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox6\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox7\s*\[\s*64\s*\]\s*=",
            r"const\s+unsigned\s+char\s+sbox8\s*\[\s*64\s*\]\s*=",
            r"void\s+des_crypt\s*\(\s*des_block_t\s*\*\s*\w+\s*,\s*des_block_t\s*\*\s*\w+\s*,\s*des_key_t\s*\*\s*\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_FLOAT64_NATIVE_DIVIDE: (
        "float64_div",
        "dfdiv.h",
        (
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+bits64\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+float64\s*;",
            r"float64\s+float64_div\s*\(\s*float64\s+\w+\s*,\s*float64\s+\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_FLOAT64_GE_PREDICATE: (
        "float64_ge",
        "float64_ge.h",
        (
            r"typedef\s+int\s+flag\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+bits64\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+float64\s*;",
            r"flag\s+float64_ge\s*\(\s*float64\s+\w+\s*,\s*float64\s+\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_FLOAT64_LE_PREDICATE: (
        "float64_le",
        "float64_le.h",
        (
            r"typedef\s+int\s+flag\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+bits64\s*;",
            r"typedef\s+unsigned\s+long\s+long(?:\s+int)?\s+float64\s*;",
            r"flag\s+float64_le\s*\(\s*float64\s+\w+\s*,\s*float64\s+\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_AES128_CIPHER: (
        "Cipher",
        "aes.h",
        (
            r"#\s*define\s+Nb\s+4\b",
            r"#\s*define\s+Nk\s+4\b",
            r"#\s*define\s+Nr\s+10\b",
            r"#\s*define\s+AES_KEYLEN\s+16\b",
            r"#\s*define\s+AES_keyExpSize\s+176\b",
            r"typedef\s+uint8_t\s+state_t\s*\[\s*4\s*\]\s*\[\s*4\s*\]\s*;",
            r"typedef\s+uint8_t\s+round_t\s*\[\s*176\s*\]\s*;",
            r"struct\s+AES_ctx\s*\{\s*uint8_t\s+RoundKey\s*\[\s*AES_keyExpSize\s*\]\s*;\s*\}\s*;",
            r"const\s+uint8_t\s+sbox\s*\[\s*256\s*\]\s*=",
            r"const\s+uint8_t\s+Rcon\s*\[\s*11\s*\]\s*=",
            r"void\s+AES_init_ctx\s*\(\s*struct\s+AES_ctx\s*\*\s*\w+\s*,\s*const\s+uint8_t\s*\*\s*\w+\s*\)\s*;",
            r"void\s+Cipher\s*\(\s*state_t\s*\*\s*\w+\s*,\s*const\s+round_t\s*\*\s*\w+\s*\)\s*;",
        ),
    ),
    CONTRACT_MD_GRID_BOUNDED_LJ: (
        "md",
        "md_grid.h",
        (
            r"#\s*define\s+TYPE\s+double\b",
            r"#\s*define\s+blockSide\s+4\b",
            r"#\s*define\s+densityFactor\s+10\b",
            r"#\s*define\s+lj1\s+1\.5\b",
            r"#\s*define\s+lj2\s+2\.0\b",
            r"typedef\s+struct\s*\{\s*TYPE\s+x\s*,\s*y\s*,\s*z\s*;\s*\}\s*dvector_t\s*;",
            r"void\s+md\s*\(\s*int32_t\s+\w+\s*\[\s*blockSide\s*\]\s*\[\s*blockSide\s*\]\s*\[\s*blockSide\s*\]",
            r"dvector_t\s+\w+\s*\[\s*blockSide\s*\]\s*\[\s*blockSide\s*\]\s*\[\s*blockSide\s*\]\s*\[\s*densityFactor\s*\]",
        ),
    ),
}


def contract_precondition_error(
    operator_id: str,
    *,
    header_name: str,
    header_text: str,
    top_fn: str,
) -> tuple[str | None, list[str]]:
    checks = [
        "exact_contract_top_function",
        "exact_immutable_header_name",
        "immutable_header_contract_tokens",
    ]
    spec = _CONTRACT_HEADERS.get(operator_id)
    if spec is None:
        return f"unsupported immutable-contract operator: {operator_id}", checks
    expected_top, expected_header, patterns = spec
    if top_fn != expected_top:
        return f"top function must be exactly {expected_top}", checks[:1]
    if header_name != expected_header:
        return f"immutable project header must be exactly {expected_header}", checks[:2]
    missing = [pattern for pattern in patterns if not re.search(pattern, header_text)]
    if missing:
        return (
            "immutable header does not prove the exact contract; missing "
            + " | ".join(missing),
            checks,
        )
    return None, checks


def _finish(
    output: str,
    actions: list[str],
    *,
    contract: str,
    bounds: str,
) -> tuple[str, list[str]]:
    actions.extend(
        [
            f"immutable_contract:{contract}",
            f"bounded_control:{bounds}",
            "reference_kernel_access:none",
        ]
    )
    return output, actions


_FUNCTION_DEFINITION = re.compile(
    r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{"
)
_CONTROL_WORDS = {"if", "for", "while", "switch", "catch"}


def _clean_contract_source(
    code: str,
    *,
    top_fn: str,
    operator_id: str,
    header_name: str,
    actions: list[str],
) -> tuple[str, list[str]]:
    """Keep one exact self-contained top definition plus its immutable header."""

    pattern = re.compile(
        rf"\b{re.escape(top_fn)}\s*\((?P<parameters>[^;{{}}]*)\)\s*\{{"
    )
    matches = list(pattern.finditer(code))
    if len(matches) != 1:
        raise ValueError(
            f"expected one definition of {top_fn} while cleaning, found {len(matches)}"
        )
    match = matches[0]
    opening = code.find("{", match.start(), match.end())
    closing = _matching_brace(code, opening)

    prefix = code[: match.start()]
    boundary = max(prefix.rfind(";"), prefix.rfind("}"))
    definition = code[boundary + 1 : closing + 1].strip()
    definition_lines = [
        line
        for line in definition.splitlines()
        if not line.lstrip().startswith("#")
    ]
    definition = "\n".join(definition_lines).strip()
    if not definition or top_fn not in definition:
        raise ValueError("failed to preserve the exact top definition")

    names = [
        item.group(1)
        for item in _FUNCTION_DEFINITION.finditer(code)
        if item.group(1) not in _CONTROL_WORDS
    ]
    removed = sum(name != top_fn for name in names)
    clean = f'#include "{header_name}"\n\n{definition}\n'
    clean_names = [
        item.group(1)
        for item in _FUNCTION_DEFINITION.finditer(clean)
        if item.group(1) not in _CONTROL_WORDS
    ]
    if clean_names != [top_fn]:
        raise ValueError(
            f"clean contract source must contain only {top_fn}, found {clean_names}"
        )
    actions.extend(
        [
            f"remove_non_top_model_helpers:{removed}",
            f"ensure_contract_header_include:{header_name}",
            f"clean_self_contained_contract_source:{operator_id}",
        ]
    )
    return clean, actions


_DES_KEY_SETUP_HELPER = r'''void des_key_setup(
    const unsigned char key[],
    des_key_t schedule,
    DES_MODE mode) {
    static const unsigned char permuted_choice_one[56] = {
        57,49,41,33,25,17,9,1,58,50,42,34,26,18,
        10,2,59,51,43,35,27,19,11,3,60,52,44,36,
        63,55,47,39,31,23,15,7,62,54,46,38,30,22,
        14,6,61,53,45,37,29,21,13,5,28,20,12,4
    };
    static const unsigned char permuted_choice_two[48] = {
        14,17,11,24,1,5,3,28,15,6,21,10,
        23,19,12,4,26,8,16,7,27,20,13,2,
        41,52,31,37,47,55,30,40,51,45,33,48,
        44,49,39,56,34,53,46,42,50,36,29,32
    };
    static const unsigned char rotations[16] = {
        1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1
    };
    unsigned long long packed_key = 0;
    for (int byte = 0; byte < 8; ++byte) {
        packed_key = (packed_key << 8) | (unsigned long long)key[byte];
    }
    unsigned long long selected_key = 0;
    for (int bit = 0; bit < 56; ++bit) {
        selected_key = (selected_key << 1) |
            ((packed_key >> (64 - permuted_choice_one[bit])) & 1ULL);
    }
    unsigned int left = (unsigned int)((selected_key >> 28) & 0x0FFFFFFFULL);
    unsigned int right = (unsigned int)(selected_key & 0x0FFFFFFFULL);
    for (int round = 0; round < 16; ++round) {
        const int shift = rotations[round];
        left = ((left << shift) | (left >> (28 - shift))) & 0x0FFFFFFFU;
        right = ((right << shift) | (right >> (28 - shift))) & 0x0FFFFFFFU;
        const unsigned long long joined =
            ((unsigned long long)left << 28) | (unsigned long long)right;
        unsigned long long round_key = 0;
        for (int bit = 0; bit < 48; ++bit) {
            round_key = (round_key << 1) |
                ((joined >> (56 - permuted_choice_two[bit])) & 1ULL);
        }
        const int destination = mode == DES_DECRYPT ? 15 - round : round;
        for (int byte = 0; byte < 6; ++byte) {
            schedule[destination][byte] =
                (unsigned char)(round_key >> (40 - 8 * byte));
        }
    }
}'''


def _inject_required_des_key_setup(
    clean: str,
    actions: list[str],
) -> tuple[str, list[str]]:
    include = '#include "des.h"\n\n'
    if not clean.startswith(include):
        raise ValueError("clean DES source must start with its exact project header")
    output = include + _DES_KEY_SETUP_HELPER + "\n\n" + clean[len(include) :]
    actions.append("emit_required_contract_helper:des_key_setup")
    return output, actions


_AES_KEY_EXPANSION_HELPERS = r'''static void KeyExpansion(
    uint8_t *RoundKey,
    const uint8_t *Key) {
    for (int word = 0; word < Nk; ++word) {
        for (int byte = 0; byte < 4; ++byte) {
            RoundKey[word * 4 + byte] = Key[word * 4 + byte];
        }
    }
    for (int word = Nk; word < Nb * (Nr + 1); ++word) {
        uint8_t temporary[4];
        for (int byte = 0; byte < 4; ++byte) {
            temporary[byte] = RoundKey[(word - 1) * 4 + byte];
        }
        if ((word % Nk) == 0) {
            const uint8_t first = temporary[0];
            temporary[0] = getSBoxValue(temporary[1]);
            temporary[1] = getSBoxValue(temporary[2]);
            temporary[2] = getSBoxValue(temporary[3]);
            temporary[3] = getSBoxValue(first);
            temporary[0] ^= Rcon[word / Nk];
        }
        for (int byte = 0; byte < 4; ++byte) {
            RoundKey[word * 4 + byte] =
                RoundKey[(word - Nk) * 4 + byte] ^ temporary[byte];
        }
    }
}

void AES_init_ctx(struct AES_ctx *ctx, const uint8_t *key) {
    KeyExpansion(ctx->RoundKey, key);
}'''


def _inject_required_aes_key_expansion(
    clean: str,
    actions: list[str],
) -> tuple[str, list[str]]:
    include = '#include "aes.h"\n\n'
    if not clean.startswith(include):
        raise ValueError("clean AES source must start with its exact project header")
    output = include + _AES_KEY_EXPANSION_HELPERS + "\n\n" + clean[len(include) :]
    actions.extend(
        [
            "emit_required_contract_helper:KeyExpansion",
            "emit_required_contract_helper:AES_init_ctx",
        ]
    )
    return output, actions


def repair_cordic_fixed_shift(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "cordic":
        raise ValueError("operator is restricted to cordic")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 3:
        raise ValueError("cordic must have theta, sine, and cosine parameters")
    theta, sine_out, cosine_out = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "const COS_SIN_TYPE inverse_gain = (COS_SIN_TYPE)0.60725293500888125617;",
            "COS_SIN_TYPE x = inverse_gain;",
            "COS_SIN_TYPE y = (COS_SIN_TYPE)0;",
            f"THETA_TYPE residual = {theta};",
            "for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {",
            "    const COS_SIN_TYPE previous_x = x;",
            "    const COS_SIN_TYPE previous_y = y;",
            "    if (residual >= (THETA_TYPE)0) {",
            "        x = previous_x - (previous_y >> iteration);",
            "        y = previous_y + (previous_x >> iteration);",
            "        residual = residual - cordic_phase[iteration];",
            "    } else {",
            "        x = previous_x + (previous_y >> iteration);",
            "        y = previous_y - (previous_x >> iteration);",
            "        residual = residual + cordic_phase[iteration];",
            "    }",
            "}",
            f"{sine_out} = y;",
            f"{cosine_out} = x;",
        ],
        CONTRACT_CORDIC_FIXED_SHIFT,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_CORDIC_FIXED_SHIFT,
        header_name="cordic.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="cordic_ap_fixed_12_2_rotation_32",
        bounds="NUM_ITERATIONS_32",
    )


def repair_float64_native_multiply(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "float64_mul":
        raise ValueError("operator is restricted to float64_mul")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 2:
        raise ValueError("float64_mul must have exactly two operands")
    left, right = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "union Float64ContractBits {",
            "    bits64 raw;",
            "    double value;",
            "};",
            f"const bits64 left_magnitude = {left} & 0x7FFFFFFFFFFFFFFFULL;",
            f"const bits64 right_magnitude = {right} & 0x7FFFFFFFFFFFFFFFULL;",
            "const bool left_is_zero = left_magnitude == 0;",
            "const bool right_is_zero = right_magnitude == 0;",
            "const bool left_is_infinity = left_magnitude == 0x7FF0000000000000ULL;",
            "const bool right_is_infinity = right_magnitude == 0x7FF0000000000000ULL;",
            "if ((left_is_infinity && right_is_zero) ||",
            "    (left_is_zero && right_is_infinity)) {",
            "    return (float64)0x7FFFFFFFFFFFFFFFULL;",
            "}",
            "Float64ContractBits left_value;",
            "Float64ContractBits right_value;",
            "Float64ContractBits product;",
            f"left_value.raw = {left};",
            f"right_value.raw = {right};",
            "product.value = left_value.value * right_value.value;",
            "return (float64)product.raw;",
        ],
        CONTRACT_FLOAT64_NATIVE_MULTIPLY,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_FLOAT64_NATIVE_MULTIPLY,
        header_name="dfmul.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="float64_bits_native_ieee754_multiply",
        bounds="single_native_multiply",
    )


def repair_needwun_row_major(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "needwun":
        raise ValueError("operator is restricted to needwun")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 6:
        raise ValueError("needwun must expose six immutable array parameters")
    seq_a, seq_b, aligned_a, aligned_b, score_matrix, pointer_matrix = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "const int match_score = 1;",
            "const int mismatch_score = -1;",
            "const int gap_score = -1;",
            "const char skip_a = 1;",
            "const char skip_b = 2;",
            "const char align = 3;",
            "const int row_stride = BLEN + 1;",
            f"{score_matrix}[0] = 0;",
            f"{pointer_matrix}[0] = 0;",
            "for (int row = 1; row <= ALEN; ++row) {",
            f"    {score_matrix}[row * row_stride] = row * gap_score;",
            f"    {pointer_matrix}[row * row_stride] = skip_b;",
            "}",
            "for (int column = 1; column <= BLEN; ++column) {",
            f"    {score_matrix}[column] = column * gap_score;",
            f"    {pointer_matrix}[column] = skip_a;",
            "}",
            "for (int row = 1; row <= ALEN; ++row) {",
            "    for (int column = 1; column <= BLEN; ++column) {",
            f"        const int substitution = ({seq_a}[row - 1] == {seq_b}[column - 1]) ? match_score : mismatch_score;",
            f"        const int from_diagonal = {score_matrix}[(row - 1) * row_stride + column - 1] + substitution;",
            f"        const int from_above = {score_matrix}[(row - 1) * row_stride + column] + gap_score;",
            f"        const int from_left = {score_matrix}[row * row_stride + column - 1] + gap_score;",
            "        int best;",
            "        char direction;",
            "        if (from_above >= from_left && from_above >= from_diagonal) {",
            "            best = from_above;",
            "            direction = skip_b;",
            "        } else if (from_left >= from_diagonal) {",
            "            best = from_left;",
            "            direction = skip_a;",
            "        } else {",
            "            best = from_diagonal;",
            "            direction = align;",
            "        }",
            f"        {score_matrix}[row * row_stride + column] = best;",
            f"        {pointer_matrix}[row * row_stride + column] = direction;",
            "    }",
            "}",
            "int row = ALEN;",
            "int column = BLEN;",
            "for (int output_index = 0; output_index < ALEN + BLEN; ++output_index) {",
            "    if (row == 0 && column == 0) {",
            f"        {aligned_a}[output_index] = '_';",
            f"        {aligned_b}[output_index] = '_';",
            "        continue;",
            "    }",
            f"    const char direction = {pointer_matrix}[row * row_stride + column];",
            "    if (row > 0 && column > 0 && direction == align) {",
            f"        {aligned_a}[output_index] = {seq_a}[row - 1];",
            f"        {aligned_b}[output_index] = {seq_b}[column - 1];",
            "        --row;",
            "        --column;",
            "    } else if (row > 0 && (column == 0 || direction == skip_b)) {",
            f"        {aligned_a}[output_index] = {seq_a}[row - 1];",
            f"        {aligned_b}[output_index] = '-';",
            "        --row;",
            "    } else {",
            f"        {aligned_a}[output_index] = '-';",
            f"        {aligned_b}[output_index] = {seq_b}[column - 1];",
            "        --column;",
            "    }",
            "}",
        ],
        CONTRACT_NEEDWUN_ROW_MAJOR_DP,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_NEEDWUN_ROW_MAJOR_DP,
        header_name="nw_nw.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="needleman_wunsch_flat_row_major_128x128",
        bounds="ALEN_128_BLEN_128_trace_256",
    )


def repair_present80_standard(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "present80_encryptBlock":
        raise ValueError("operator is restricted to present80_encryptBlock")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 2:
        raise ValueError("present80_encryptBlock must have block and key pointers")
    block, key = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "unsigned char working_key[PRESENT_80_KEY_SIZE_BYTES];",
            "for (int byte = 0; byte < PRESENT_80_KEY_SIZE_BYTES; ++byte) {",
            f"    working_key[byte] = (*{key})[byte];",
            "}",
            "unsigned long long state_value = 0;",
            "for (int byte = 0; byte < PRESENT_BLOCK_SIZE_BYTES; ++byte) {",
            f"    state_value = (state_value << 8) | (unsigned long long)(*{block})[byte];",
            "}",
            "for (int round = 1; round < ROUNDS; ++round) {",
            "    unsigned long long round_key = 0;",
            "    for (int byte = 0; byte < ROUND_KEY_SIZE_BYTES; ++byte) {",
            "        round_key = (round_key << 8) | (unsigned long long)working_key[byte];",
            "    }",
            "    state_value ^= round_key;",
            "    unsigned long long substituted = 0;",
            "    for (int nibble = 0; nibble < 16; ++nibble) {",
            "        const int shift = nibble * 4;",
            "        const unsigned char value = (unsigned char)((state_value >> shift) & 0xFULL);",
            "        substituted |= ((unsigned long long)sBox[value]) << shift;",
            "    }",
            "    unsigned long long permuted = substituted & (1ULL << 63);",
            "    for (int bit = 0; bit < 63; ++bit) {",
            "        const int destination = (16 * bit) % 63;",
            "        permuted |= ((substituted >> bit) & 1ULL) << destination;",
            "    }",
            "    state_value = permuted;",
            "    unsigned char rotated[PRESENT_80_KEY_SIZE_BYTES];",
            "    for (int byte = 0; byte < PRESENT_80_KEY_SIZE_BYTES; ++byte) {",
            "        rotated[byte] = 0;",
            "    }",
            "    for (int destination = 0; destination < 80; ++destination) {",
            "        const int source = (destination + 61) % 80;",
            "        const unsigned char bit = (unsigned char)((working_key[source / 8] >> (7 - source % 8)) & 1U);",
            "        rotated[destination / 8] |= (unsigned char)(bit << (7 - destination % 8));",
            "    }",
            "    for (int byte = 0; byte < PRESENT_80_KEY_SIZE_BYTES; ++byte) {",
            "        working_key[byte] = rotated[byte];",
            "    }",
            "    working_key[0] = (unsigned char)((sBox[working_key[0] >> 4] << 4) | (working_key[0] & 0x0F));",
            "    working_key[7] ^= (unsigned char)(round >> 1);",
            "    working_key[8] ^= (unsigned char)(round << 7);",
            "}",
            "unsigned long long final_round_key = 0;",
            "for (int byte = 0; byte < ROUND_KEY_SIZE_BYTES; ++byte) {",
            "    final_round_key = (final_round_key << 8) | (unsigned long long)working_key[byte];",
            "}",
            "state_value ^= final_round_key;",
            "for (int byte = 0; byte < PRESENT_BLOCK_SIZE_BYTES; ++byte) {",
            f"    (*{block})[byte] = (unsigned char)(state_value >> (56 - 8 * byte));",
            "}",
        ],
        CONTRACT_PRESENT80_STANDARD,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_PRESENT80_STANDARD,
        header_name="present.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="present80_big_endian_31_rounds_plus_final_key",
        bounds="ROUNDS_32_key_bits_80_state_bits_64",
    )


def repair_des_feistel_schedule(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "des_crypt":
        raise ValueError("operator is restricted to des_crypt")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 3:
        raise ValueError("des_crypt must have input, output, and schedule pointers")
    input_block, output_block, schedule = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "static const unsigned char initial_permutation[64] = {",
            "    58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,",
            "    62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,",
            "    57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,",
            "    61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7",
            "};",
            "static const unsigned char final_permutation[64] = {",
            "    40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,",
            "    38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,",
            "    36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,",
            "    34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25",
            "};",
            "static const unsigned char expansion[48] = {",
            "    32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,",
            "    12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,",
            "    24,25,26,27,28,29,28,29,30,31,32,1",
            "};",
            "static const unsigned char p_permutation[32] = {",
            "    16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,",
            "    2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25",
            "};",
            "unsigned long long packed_input = 0;",
            "for (int byte = 0; byte < 8; ++byte) {",
            f"    packed_input = (packed_input << 8) | (unsigned long long)(*{input_block})[byte];",
            "}",
            "unsigned long long permuted_input = 0;",
            "for (int bit = 0; bit < 64; ++bit) {",
            "    permuted_input = (permuted_input << 1) | ((packed_input >> (64 - initial_permutation[bit])) & 1ULL);",
            "}",
            "unsigned int left = (unsigned int)(permuted_input >> 32);",
            "unsigned int right = (unsigned int)permuted_input;",
            "for (int round = 0; round < 16; ++round) {",
            "    unsigned long long expanded = 0;",
            "    for (int bit = 0; bit < 48; ++bit) {",
            "        expanded = (expanded << 1) | ((right >> (32 - expansion[bit])) & 1U);",
            "    }",
            "    unsigned long long round_key = 0;",
            "    for (int byte = 0; byte < 6; ++byte) {",
            f"        round_key = (round_key << 8) | (unsigned long long)(*{schedule})[round][byte];",
            "    }",
            "    const unsigned long long mixed = expanded ^ round_key;",
            "    const unsigned int substituted =",
            "        ((unsigned int)sbox1[SBOXBIT((unsigned char)(mixed >> 42))] << 28) |",
            "        ((unsigned int)sbox2[SBOXBIT((unsigned char)(mixed >> 36))] << 24) |",
            "        ((unsigned int)sbox3[SBOXBIT((unsigned char)(mixed >> 30))] << 20) |",
            "        ((unsigned int)sbox4[SBOXBIT((unsigned char)(mixed >> 24))] << 16) |",
            "        ((unsigned int)sbox5[SBOXBIT((unsigned char)(mixed >> 18))] << 12) |",
            "        ((unsigned int)sbox6[SBOXBIT((unsigned char)(mixed >> 12))] << 8) |",
            "        ((unsigned int)sbox7[SBOXBIT((unsigned char)(mixed >> 6))] << 4) |",
            "        ((unsigned int)sbox8[SBOXBIT((unsigned char)mixed)]);",
            "    unsigned int feistel = 0;",
            "    for (int bit = 0; bit < 32; ++bit) {",
            "        feistel = (feistel << 1) | ((substituted >> (32 - p_permutation[bit])) & 1U);",
            "    }",
            "    const unsigned int next_left = right;",
            "    right = left ^ feistel;",
            "    left = next_left;",
            "}",
            "const unsigned long long preoutput = ((unsigned long long)right << 32) | left;",
            "unsigned long long packed_output = 0;",
            "for (int bit = 0; bit < 64; ++bit) {",
            "    packed_output = (packed_output << 1) | ((preoutput >> (64 - final_permutation[bit])) & 1ULL);",
            "}",
            "for (int byte = 0; byte < 8; ++byte) {",
            f"    (*{output_block})[byte] = (unsigned char)(packed_output >> (56 - 8 * byte));",
            "}",
        ],
        CONTRACT_DES_FEISTEL_SCHEDULE,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_DES_FEISTEL_SCHEDULE,
        header_name="des.h",
        actions=actions,
    )
    output, actions = _inject_required_des_key_setup(output, actions)
    return _finish(
        output,
        actions,
        contract="des_standard_feistel_supplied_16x6_schedule",
        bounds="rounds_16_block_bits_64_subkey_bits_48",
    )


def repair_float64_native_divide(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "float64_div":
        raise ValueError("operator is restricted to float64_div")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 2:
        raise ValueError("float64_div must have exactly two operands")
    dividend, divisor = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "union Float64DivideContractBits {",
            "    bits64 raw;",
            "    double value;",
            "};",
            f"const bits64 dividend_magnitude = {dividend} & 0x7FFFFFFFFFFFFFFFULL;",
            f"const bits64 divisor_magnitude = {divisor} & 0x7FFFFFFFFFFFFFFFULL;",
            "const bool dividend_is_nan = dividend_magnitude > 0x7FF0000000000000ULL;",
            "const bool divisor_is_nan = divisor_magnitude > 0x7FF0000000000000ULL;",
            "const bool dividend_is_infinity = dividend_magnitude == 0x7FF0000000000000ULL;",
            "const bool divisor_is_infinity = divisor_magnitude == 0x7FF0000000000000ULL;",
            "const bool dividend_is_zero = dividend_magnitude == 0;",
            "const bool divisor_is_zero = divisor_magnitude == 0;",
            f"if (dividend_is_nan) return (float64){dividend};",
            f"if (divisor_is_nan) return (float64){divisor};",
            "if ((dividend_is_infinity && divisor_is_infinity) ||",
            "    (dividend_is_zero && divisor_is_zero)) {",
            "    return (float64)0x7FFFFFFFFFFFFFFFULL;",
            "}",
            "Float64DivideContractBits dividend_value;",
            "Float64DivideContractBits divisor_value;",
            "Float64DivideContractBits quotient;",
            f"dividend_value.raw = {dividend};",
            f"divisor_value.raw = {divisor};",
            "quotient.value = dividend_value.value / divisor_value.value;",
            "return (float64)quotient.raw;",
        ],
        CONTRACT_FLOAT64_NATIVE_DIVIDE,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_FLOAT64_NATIVE_DIVIDE,
        header_name="dfdiv.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="float64_bits_native_ieee754_divide",
        bounds="single_native_divide_with_bounded_special_cases",
    )


def _repair_float64_contract_predicate(
    code: str,
    top_fn: str,
    *,
    operator_id: str,
    header_name: str,
) -> tuple[str, list[str]]:
    output, actions = repair_ieee754_predicate(code, top_fn)
    expected_action = f"replace_top_body:ieee754_predicate:{top_fn}"
    if not actions or actions[0] != expected_action:
        raise ValueError("IEEE-754 predicate did not emit its exact replacement audit")
    actions[0] = f"replace_top_body:{operator_id}:{top_fn}"
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=operator_id,
        header_name=header_name,
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract=f"{top_fn}_ieee754_binary64_ordering",
        bounds="single_bounded_binary64_predicate",
    )


def repair_float64_ge_contract(
    code: str, top_fn: str
) -> tuple[str, list[str]]:
    if top_fn != "float64_ge":
        raise ValueError("operator is restricted to float64_ge")
    return _repair_float64_contract_predicate(
        code,
        top_fn,
        operator_id=CONTRACT_FLOAT64_GE_PREDICATE,
        header_name="float64_ge.h",
    )


def repair_float64_le_contract(
    code: str, top_fn: str
) -> tuple[str, list[str]]:
    if top_fn != "float64_le":
        raise ValueError("operator is restricted to float64_le")
    return _repair_float64_contract_predicate(
        code,
        top_fn,
        operator_id=CONTRACT_FLOAT64_LE_PREDICATE,
        header_name="float64_le.h",
    )


def repair_aes128_cipher(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "Cipher":
        raise ValueError("operator is restricted to Cipher")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 2:
        raise ValueError("Cipher must have state and expanded-round-key parameters")
    state, round_key = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "for (int column = 0; column < Nb; ++column) {",
            "    for (int row = 0; row < 4; ++row) {",
            f"        (*{state})[column][row] ^= (*{round_key})[column * 4 + row];",
            "    }",
            "}",
            "for (int round = 1; round < Nr; ++round) {",
            "    for (int column = 0; column < 4; ++column) {",
            "        for (int row = 0; row < 4; ++row) {",
            f"            (*{state})[column][row] = getSBoxValue((*{state})[column][row]);",
            "        }",
            "    }",
            f"    uint8_t temporary = (*{state})[0][1];",
            f"    (*{state})[0][1] = (*{state})[1][1];",
            f"    (*{state})[1][1] = (*{state})[2][1];",
            f"    (*{state})[2][1] = (*{state})[3][1];",
            f"    (*{state})[3][1] = temporary;",
            f"    temporary = (*{state})[0][2];",
            f"    (*{state})[0][2] = (*{state})[2][2];",
            f"    (*{state})[2][2] = temporary;",
            f"    temporary = (*{state})[1][2];",
            f"    (*{state})[1][2] = (*{state})[3][2];",
            f"    (*{state})[3][2] = temporary;",
            f"    temporary = (*{state})[3][3];",
            f"    (*{state})[3][3] = (*{state})[2][3];",
            f"    (*{state})[2][3] = (*{state})[1][3];",
            f"    (*{state})[1][3] = (*{state})[0][3];",
            f"    (*{state})[0][3] = temporary;",
            "    for (int column = 0; column < 4; ++column) {",
            f"        const uint8_t first = (*{state})[column][0];",
            f"        const uint8_t all = (*{state})[column][0] ^ (*{state})[column][1] ^",
            f"            (*{state})[column][2] ^ (*{state})[column][3];",
            f"        uint8_t pair = (*{state})[column][0] ^ (*{state})[column][1];",
            "        uint8_t doubled = (uint8_t)((pair << 1) ^ (((pair >> 7) & 1U) * 0x1BU));",
            f"        (*{state})[column][0] ^= doubled ^ all;",
            f"        pair = (*{state})[column][1] ^ (*{state})[column][2];",
            "        doubled = (uint8_t)((pair << 1) ^ (((pair >> 7) & 1U) * 0x1BU));",
            f"        (*{state})[column][1] ^= doubled ^ all;",
            f"        pair = (*{state})[column][2] ^ (*{state})[column][3];",
            "        doubled = (uint8_t)((pair << 1) ^ (((pair >> 7) & 1U) * 0x1BU));",
            f"        (*{state})[column][2] ^= doubled ^ all;",
            f"        pair = (*{state})[column][3] ^ first;",
            "        doubled = (uint8_t)((pair << 1) ^ (((pair >> 7) & 1U) * 0x1BU));",
            f"        (*{state})[column][3] ^= doubled ^ all;",
            "    }",
            "    for (int column = 0; column < Nb; ++column) {",
            "        for (int row = 0; row < 4; ++row) {",
            f"            (*{state})[column][row] ^= (*{round_key})[round * Nb * 4 + column * 4 + row];",
            "        }",
            "    }",
            "}",
            "for (int column = 0; column < 4; ++column) {",
            "    for (int row = 0; row < 4; ++row) {",
            f"        (*{state})[column][row] = getSBoxValue((*{state})[column][row]);",
            "    }",
            "}",
            f"uint8_t temporary = (*{state})[0][1];",
            f"(*{state})[0][1] = (*{state})[1][1];",
            f"(*{state})[1][1] = (*{state})[2][1];",
            f"(*{state})[2][1] = (*{state})[3][1];",
            f"(*{state})[3][1] = temporary;",
            f"temporary = (*{state})[0][2];",
            f"(*{state})[0][2] = (*{state})[2][2];",
            f"(*{state})[2][2] = temporary;",
            f"temporary = (*{state})[1][2];",
            f"(*{state})[1][2] = (*{state})[3][2];",
            f"(*{state})[3][2] = temporary;",
            f"temporary = (*{state})[3][3];",
            f"(*{state})[3][3] = (*{state})[2][3];",
            f"(*{state})[2][3] = (*{state})[1][3];",
            f"(*{state})[1][3] = (*{state})[0][3];",
            f"(*{state})[0][3] = temporary;",
            "for (int column = 0; column < Nb; ++column) {",
            "    for (int row = 0; row < 4; ++row) {",
            f"        (*{state})[column][row] ^= (*{round_key})[Nr * Nb * 4 + column * 4 + row];",
            "    }",
            "}",
        ],
        CONTRACT_AES128_CIPHER,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_AES128_CIPHER,
        header_name="aes.h",
        actions=actions,
    )
    output, actions = _inject_required_aes_key_expansion(output, actions)
    return _finish(
        output,
        actions,
        contract="aes128_fips197_cipher_with_176_byte_key_schedule",
        bounds="Nr_10_rounds_state_4x4_key_expansion_44_words",
    )


def repair_md_grid_bounded_lj(code: str, top_fn: str) -> tuple[str, list[str]]:
    if top_fn != "md":
        raise ValueError("operator is restricted to md")
    parameters = _top_parameter_names(code, top_fn)
    if len(parameters) != 3:
        raise ValueError("md must have n_points, force, and position parameters")
    n_points, force, position = parameters
    output, actions = _replace_top_body(
        code,
        top_fn,
        [
            "for (int block_x = 0; block_x < blockSide; ++block_x) {",
            "    for (int block_y = 0; block_y < blockSide; ++block_y) {",
            "        for (int block_z = 0; block_z < blockSide; ++block_z) {",
            f"            const int point_count = {n_points}[block_x][block_y][block_z];",
            "            for (int point = 0; point < densityFactor; ++point) {",
            "                if (point >= point_count) {",
            "                    continue;",
            "                }",
            f"                const TYPE point_x = {position}[block_x][block_y][block_z][point].x;",
            f"                const TYPE point_y = {position}[block_x][block_y][block_z][point].y;",
            f"                const TYPE point_z = {position}[block_x][block_y][block_z][point].z;",
            "                TYPE force_x = (TYPE)0;",
            "                TYPE force_y = (TYPE)0;",
            "                TYPE force_z = (TYPE)0;",
            "                for (int offset_x = -1; offset_x <= 1; ++offset_x) {",
            "                    const int neighbor_x = block_x + offset_x;",
            "                    if (neighbor_x < 0 || neighbor_x >= blockSide) continue;",
            "                    for (int offset_y = -1; offset_y <= 1; ++offset_y) {",
            "                        const int neighbor_y = block_y + offset_y;",
            "                        if (neighbor_y < 0 || neighbor_y >= blockSide) continue;",
            "                        for (int offset_z = -1; offset_z <= 1; ++offset_z) {",
            "                            const int neighbor_z = block_z + offset_z;",
            "                            if (neighbor_z < 0 || neighbor_z >= blockSide) continue;",
            f"                            const int neighbor_count = {n_points}[neighbor_x][neighbor_y][neighbor_z];",
            "                            for (int neighbor = 0; neighbor < densityFactor; ++neighbor) {",
            "                                if (neighbor >= neighbor_count) continue;",
            f"                                const TYPE delta_x = point_x - {position}[neighbor_x][neighbor_y][neighbor_z][neighbor].x;",
            f"                                const TYPE delta_y = point_y - {position}[neighbor_x][neighbor_y][neighbor_z][neighbor].y;",
            f"                                const TYPE delta_z = point_z - {position}[neighbor_x][neighbor_y][neighbor_z][neighbor].z;",
            "                                const TYPE distance_squared =",
            "                                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;",
            "                                if (distance_squared != (TYPE)0) {",
            "                                    const TYPE inverse_squared = (TYPE)1 / distance_squared;",
            "                                    const TYPE inverse_sixth =",
            "                                        inverse_squared * inverse_squared * inverse_squared;",
            "                                    const TYPE potential =",
            "                                        inverse_sixth * ((TYPE)lj1 * inverse_sixth - (TYPE)lj2);",
            "                                    const TYPE scalar_force = inverse_squared * potential;",
            "                                    force_x += delta_x * scalar_force;",
            "                                    force_y += delta_y * scalar_force;",
            "                                    force_z += delta_z * scalar_force;",
            "                                }",
            "                            }",
            "                        }",
            "                    }",
            "                }",
            f"                {force}[block_x][block_y][block_z][point].x = force_x;",
            f"                {force}[block_x][block_y][block_z][point].y = force_y;",
            f"                {force}[block_x][block_y][block_z][point].z = force_z;",
            "            }",
            "        }",
            "    }",
            "}",
        ],
        CONTRACT_MD_GRID_BOUNDED_LJ,
    )
    output, actions = _clean_contract_source(
        output,
        top_fn=top_fn,
        operator_id=CONTRACT_MD_GRID_BOUNDED_LJ,
        header_name="md_grid.h",
        actions=actions,
    )
    return _finish(
        output,
        actions,
        contract="md_grid_4x4x4_density10_neighbor_lennard_jones",
        bounds="blocks_64_points_10_neighbors_27_neighbor_points_10",
    )


_CONTRACT_OPERATORS: dict[str, Callable[[str, str], tuple[str, list[str]]]] = {
    CONTRACT_CORDIC_FIXED_SHIFT: repair_cordic_fixed_shift,
    CONTRACT_FLOAT64_NATIVE_MULTIPLY: repair_float64_native_multiply,
    CONTRACT_NEEDWUN_ROW_MAJOR_DP: repair_needwun_row_major,
    CONTRACT_PRESENT80_STANDARD: repair_present80_standard,
    CONTRACT_DES_FEISTEL_SCHEDULE: repair_des_feistel_schedule,
    CONTRACT_FLOAT64_NATIVE_DIVIDE: repair_float64_native_divide,
    CONTRACT_AES128_CIPHER: repair_aes128_cipher,
    CONTRACT_MD_GRID_BOUNDED_LJ: repair_md_grid_bounded_lj,
    CONTRACT_FLOAT64_GE_PREDICATE: repair_float64_ge_contract,
    CONTRACT_FLOAT64_LE_PREDICATE: repair_float64_le_contract,
}


def apply_contract_operator(
    operator_id: str,
    code: str,
    top_fn: str,
) -> tuple[str, list[str]]:
    operator = _CONTRACT_OPERATORS.get(operator_id)
    if operator is None:
        raise ValueError(f"unsupported immutable-contract operator: {operator_id}")
    return operator(code, top_fn)
