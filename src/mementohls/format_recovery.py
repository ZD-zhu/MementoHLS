from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_OPEN = re.compile(r'<OUTPUT_CODE\s+name="([^"]+)">')
_CLOSE = re.compile(r'</OUTPUT_CODE(?:\s+name="([^"]+)")?>')
_FENCE = re.compile(
    r'(?ms)^[ \t]*```(?P<language>cpp|c\+\+|cc|cxx)[ \t]*\r?\n'
    r'(?P<body>.*?)^[ \t]*```[ \t]*(?:\r?\n|\Z)',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FormatRecoveryResult:
    enabled: bool
    checked: bool
    attempted: bool
    accepted: bool
    kind: str
    reason: str
    finish_reason: str | None
    raw_response_sha256: str
    effective_response_sha256: str
    body_sha256_before: str | None
    body_sha256_after: str | None
    body_sha256_preserved: bool
    semantic_candidate_count: int
    additional_llm_calls: int
    additional_reviewer_calls: int
    effective_response: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("effective_response")
        return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _without_comments_and_literals(text: str) -> str:
    output: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if char == "/" and nxt == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if char in {'"', "'"}:
                quote = char
                output.append(" ")
                index += 1
                state = "literal"
                continue
            output.append(char)
            index += 1
            continue
        if state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "code"
            continue
        if state == "block_comment":
            if char == "*" and nxt == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if char == "\\":
            output.append(" ")
            if index + 1 < len(text):
                output.append("\n" if text[index + 1] == "\n" else " ")
            index += 2
            continue
        output.append("\n" if char == "\n" else " ")
        index += 1
        if char == quote:
            state = "code"
    return "".join(output)


def _is_complete_cpp_body(body: str, top_fn: str) -> tuple[bool, str]:
    if not body.strip():
        return False, "empty_body"
    sanitized = _without_comments_and_literals(body)
    if not re.search(
        rf"\b{re.escape(top_fn)}\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{",
        sanitized,
        re.MULTILINE,
    ):
        return False, "top_function_definition_missing"
    depth = 0
    saw_brace = False
    for char in sanitized:
        if char == "{":
            depth += 1
            saw_brace = True
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False, "closing_brace_without_opening_brace"
    if not saw_brace or depth != 0:
        return False, "unbalanced_braces"
    stripped = sanitized.rstrip()
    if not (stripped.endswith("}") or stripped.endswith(";") or re.search(r"#\s*endif\s*$", stripped)):
        return False, "trailing_non_cpp_text_or_truncation"
    return True, "complete"


def _result(
    raw: str,
    effective: str,
    *,
    enabled: bool,
    checked: bool,
    attempted: bool,
    accepted: bool,
    kind: str,
    reason: str,
    finish_reason: str | None,
    body_before: str | None = None,
    body_after: str | None = None,
) -> FormatRecoveryResult:
    before_sha = _sha256(body_before) if body_before is not None else None
    after_sha = _sha256(body_after) if body_after is not None else None
    return FormatRecoveryResult(
        enabled=enabled,
        checked=checked,
        attempted=attempted,
        accepted=accepted,
        kind=kind,
        reason=reason,
        finish_reason=finish_reason,
        raw_response_sha256=_sha256(raw),
        effective_response_sha256=_sha256(effective),
        body_sha256_before=before_sha,
        body_sha256_after=after_sha,
        body_sha256_preserved=(
            before_sha is not None and before_sha == after_sha
        ),
        semantic_candidate_count=1,
        additional_llm_calls=0,
        additional_reviewer_calls=0,
        effective_response=effective,
    )


def recover_hls_envelope(
    raw_response: str,
    *,
    expected_filename: str,
    top_fn: str,
    finish_reason: str | None,
    enabled: bool,
) -> FormatRecoveryResult:
    """Canonicalize only a provably unique complete C++ body; never regenerate code."""
    if not enabled:
        return _result(
            raw_response,
            raw_response,
            enabled=False,
            checked=False,
            attempted=False,
            accepted=False,
            kind="disabled",
            reason="mode_does_not_use_rfl",
            finish_reason=finish_reason,
        )

    openings = list(_OPEN.finditer(raw_response))
    closings = list(_CLOSE.finditer(raw_response))
    fences = list(_FENCE.finditer(raw_response))
    fence_markers = raw_response.count("```")
    if (
        len(openings) == 1
        and len(closings) == 1
        and openings[0].end() <= closings[0].start()
        and (
            closings[0].group(1) is None
            or closings[0].group(1) == openings[0].group(1)
        )
    ):
        return _result(
            raw_response,
            raw_response,
            enabled=True,
            checked=True,
            attempted=False,
            accepted=False,
            kind="strict_xml_already_present",
            reason="no_recovery_needed",
            finish_reason=finish_reason,
        )

    if finish_reason != "stop":
        return _result(
            raw_response,
            raw_response,
            enabled=True,
            checked=True,
            attempted=True,
            accepted=False,
            kind="rejected_non_stop_finish",
            reason="length_or_unknown_finish_cannot_prove_complete_body",
            finish_reason=finish_reason,
        )

    if openings or closings:
        if fences or fence_markers:
            return _result(
                raw_response,
                raw_response,
                enabled=True,
                checked=True,
                attempted=True,
                accepted=False,
                kind="rejected_mixed_envelopes",
                reason="xml_and_markdown_fence_are_both_present",
                finish_reason=finish_reason,
            )
        if len(openings) != 1 or closings:
            return _result(
                raw_response,
                raw_response,
                enabled=True,
                checked=True,
                attempted=True,
                accepted=False,
                kind="rejected_ambiguous_xml",
                reason="requires_exactly_one_opening_and_zero_closing_tags",
                finish_reason=finish_reason,
            )
        opening = openings[0]
        if raw_response[: opening.start()].strip():
            return _result(
                raw_response,
                raw_response,
                enabled=True,
                checked=True,
                attempted=True,
                accepted=False,
                kind="rejected_prefix_prose",
                reason="single_opening_recovery_requires_whitespace_only_before_xml",
                finish_reason=finish_reason,
            )
        body = raw_response[opening.end() :]
        complete, reason = _is_complete_cpp_body(body, top_fn)
        if not complete:
            return _result(
                raw_response,
                raw_response,
                enabled=True,
                checked=True,
                attempted=True,
                accepted=False,
                kind="rejected_incomplete_open_xml_body",
                reason=reason,
                finish_reason=finish_reason,
                body_before=body,
                body_after=body,
            )
        effective = raw_response + "</OUTPUT_CODE>"
        return _result(
            raw_response,
            effective,
            enabled=True,
            checked=True,
            attempted=True,
            accepted=True,
            kind="append_missing_closing_tag",
            reason="unique_complete_body_preserved_byte_for_byte",
            finish_reason=finish_reason,
            body_before=body,
            body_after=body,
        )

    if fence_markers != 2 or len(fences) != 1:
        return _result(
            raw_response,
            raw_response,
            enabled=True,
            checked=True,
            attempted=True,
            accepted=False,
            kind="rejected_ambiguous_or_plain_text",
            reason="requires_exactly_one_labeled_complete_cpp_fence",
            finish_reason=finish_reason,
        )
    body = fences[0].group("body")
    complete, reason = _is_complete_cpp_body(body, top_fn)
    if not complete:
        return _result(
            raw_response,
            raw_response,
            enabled=True,
            checked=True,
            attempted=True,
            accepted=False,
            kind="rejected_incomplete_fenced_body",
            reason=reason,
            finish_reason=finish_reason,
            body_before=body,
            body_after=body,
        )
    safe_filename = Path(expected_filename).name
    effective = f'<OUTPUT_CODE name="{safe_filename}">{body}</OUTPUT_CODE>'
    return _result(
        raw_response,
        effective,
        enabled=True,
        checked=True,
        attempted=True,
        accepted=True,
        kind="wrap_unique_labeled_cpp_fence",
        reason="unique_complete_body_preserved_byte_for_byte",
        finish_reason=finish_reason,
        body_before=body,
        body_after=body,
    )
