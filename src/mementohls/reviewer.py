from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .compat import enable_python310_hls_eval
from .datasets import (
    DATASET_BENCH4HLS,
    DATASET_HLS_EVAL,
    case_expected_filename,
)
from .deterministic_csim import DeterministicVitisHLSCSimTool
from .models import FailureStage, ReviewResult, atomic_write_text


_OUTPUT_OPEN_TAG = re.compile(r'<OUTPUT_CODE\s+name="([^"]+)">')
_OUTPUT_CLOSE_TAG = re.compile(r'</OUTPUT_CODE(?:\s+name="([^"]+)")?>')

# These signatures deliberately require a strong tool/OS failure marker. Generic
# words such as "error", a non-zero return code, and a design timeout are not
# infrastructure failures and must remain attributable to the generated design.
_INFRASTRUCTURE_LOG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "license",
        re.compile(
            r"(?:license checkout failed|failed to check\s*out[^\n]*license|"
            r"unable to (?:obtain|check\s*out)[^\n]*license|no valid license|"
            r"a valid license was not found|license server machine is down)",
            re.IGNORECASE,
        ),
    ),
    (
        "internal_tool_error",
        re.compile(
            r"(?:\binternal compiler error\b|\bINTERNAL_ERROR\b|"
            r"\b(?:Vitis(?: HLS)?|Vivado|Xilinx|LLVM|clang)[^\n]{0,80}"
            r"\binternal error\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "storage_exhausted",
        re.compile(r"(?:no space left on device|disk quota exceeded)", re.IGNORECASE),
    ),
    (
        "filesystem_io",
        re.compile(
            r"(?:input/output error|\bI/O error\b|stale file handle|"
            r"read-only file system)",
            re.IGNORECASE,
        ),
    ),
)


def _execution_dict(output: Any | None) -> dict[str, Any] | None:
    if output is None:
        return None
    execution = output.data_execution
    return {
        "return_code": execution.return_code,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
        "t0": execution.t0,
        "t1": execution.t1,
        "execution_time": execution.execution_time,
        "timeout": execution.timeout,
    }


def failure_log(review: ReviewResult) -> str:
    if review.failure_stage == FailureStage.PARSE.value:
        return review.parse_error or "The model output did not satisfy the OUTPUT_CODE XML contract."
    if review.failure_stage == FailureStage.COMPILE.value:
        payload = review.compile or {}
    elif review.failure_stage == FailureStage.TB.value:
        payload = review.testbench or {}
        oracle = payload.get("oracle") or {}
        if oracle.get("kind") == "bench4hls_stdout_marker_v1":
            return (
                "BENCH4HLS_TB_RESULT: "
                f"passed={bool(oracle.get('passed'))}; "
                f"success_marker_count={int(oracle.get('success_marker_count') or 0)}; "
                f"failure_marker_count={int(oracle.get('failure_marker_count') or 0)}; "
                f"return_code={payload.get('return_code')}; "
                f"timeout={bool(payload.get('timeout'))}. "
                "The private testbench output is intentionally withheld."
            )
    else:
        payload = review.synthesis or {}
    lines: list[str] = []
    if payload.get("timeout"):
        lines.append(
            "KERNELMEM_TOOL_TIMEOUT: "
            f"stage={review.failure_stage}; timeout=true; "
            f"return_code={payload.get('return_code')}"
        )
    lines.extend(str(payload[key]) for key in ("stderr", "stdout") if payload.get(key))
    return "\n".join(lines)


def classify_infrastructure_failure(
    executions: dict[str, dict[str, Any] | None],
) -> str | None:
    """Return a conservative infrastructure classification for failed tool logs."""

    for phase, execution in executions.items():
        if not execution or execution.get("return_code") in (None, 0):
            continue
        log = "\n".join(
            str(execution.get(key, "")) for key in ("stderr", "stdout") if execution.get(key)
        )
        for category, pattern in _INFRASTRUCTURE_LOG_PATTERNS:
            match = pattern.search(log)
            if match:
                evidence = " ".join(match.group(0).split())[:240]
                return f"{phase} infrastructure log [{category}]: {evidence}"
    return None


def _append_infrastructure_error(result: ReviewResult, error: str | None) -> None:
    if not error:
        return
    if result.infrastructure_error:
        result.infrastructure_error = f"{result.infrastructure_error}; {error}"
    else:
        result.infrastructure_error = error


def bench4hls_tb_oracle(
    execution: dict[str, Any] | None,
    *,
    frozen_epoch: str,
) -> dict[str, Any]:
    payload = execution or {}
    stdout = str(payload.get("stdout") or "")
    success_count = len(re.findall(r"\bTest Passed\b", stdout, re.IGNORECASE))
    failure_count = len(re.findall(r"\bTest Failed\b", stdout, re.IGNORECASE))
    passed = bool(
        payload
        and payload.get("return_code") == 0
        and not payload.get("timeout")
        and success_count >= 1
        and failure_count == 0
    )
    return {
        "kind": "bench4hls_stdout_marker_v1",
        "passed": passed,
        "success_marker_count": success_count,
        "failure_marker_count": failure_count,
        "frozen_epoch": frozen_epoch,
    }


class ToolPools:
    def __init__(self, csim_workers: int, synth_workers: int) -> None:
        self.csim = ThreadPoolExecutor(max_workers=csim_workers, thread_name_prefix="hls-csim")
        self.synth = ThreadPoolExecutor(max_workers=synth_workers, thread_name_prefix="hls-synth")

    def shutdown(self) -> None:
        self.csim.shutdown(wait=True)
        self.synth.shutdown(wait=True)


class HLSReviewer:
    def __init__(
        self,
        hls_eval_root: Path,
        vitis_dir: Path,
        fpga_part: str,
        clock_ns: float,
        csim_timeout: float,
        synth_timeout: float,
        pools: ToolPools,
        dataset_kind: str = DATASET_HLS_EVAL,
        bench4hls_frozen_epoch: str = "@2027-01-01 00:00:00",
    ) -> None:
        enable_python310_hls_eval(hls_eval_root)
        from hls_eval.prompting import extract_code_xml_from_llm_output
        from hls_eval.tools import VitisHLSCSimTool, VitisHLSSynthTool

        if dataset_kind not in {DATASET_HLS_EVAL, DATASET_BENCH4HLS}:
            raise ValueError(f"Unsupported reviewer dataset: {dataset_kind}")
        self.dataset_kind = dataset_kind
        self.bench4hls_frozen_epoch = bench4hls_frozen_epoch
        self.extract_code = extract_code_xml_from_llm_output
        self.csim_tool = (
            DeterministicVitisHLSCSimTool(
                vitis_dir,
                frozen_epoch=bench4hls_frozen_epoch,
            )
            if dataset_kind == DATASET_BENCH4HLS
            else VitisHLSCSimTool(vitis_dir)
        )
        self.synth_tool = VitisHLSSynthTool(vitis_dir)
        self.fpga_part = fpga_part
        self.clock_ns = clock_ns
        self.csim_total_timeout = csim_timeout
        # HLS-Eval applies its timeout once to compile setup and once to the TB
        # executable. Split the frozen 360 s budget so the combined C-sim path
        # remains bounded by 360 s rather than silently reaching 720 s.
        self.csim_timeout = csim_timeout / 2.0
        self.synth_timeout = synth_timeout
        self.pools = pools

    @staticmethod
    def infer_expected_filename(benchmark_case: Any) -> str:
        """Infer a deterministic design basename from immutable task metadata."""

        explicit = getattr(benchmark_case, "expected_filename", None)
        if isinstance(explicit, str) and explicit:
            return case_expected_filename(benchmark_case)

        headers = sorted((Path(path) for path in benchmark_case.h_files), key=lambda p: p.name)
        if len(headers) == 1:
            return f"{headers[0].stem}.cpp"

        preferred_stems: list[str] = []
        case_name = str(getattr(benchmark_case, "name", "") or "")
        if case_name:
            preferred_stems.append(Path(case_name).name)
        tb_file = getattr(benchmark_case, "tb_file", None)
        if tb_file is not None:
            tb_stem = Path(tb_file).stem
            preferred_stems.append(
                tb_stem[:-3] if tb_stem.lower().endswith("_tb") else tb_stem
            )
        preferred_stems.append(str(getattr(benchmark_case, "top_fn", "") or ""))

        for preferred in preferred_stems:
            matches = [header for header in headers if header.stem == preferred]
            if len(matches) == 1:
                return f"{matches[0].stem}.cpp"

        top_fn = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            str(getattr(benchmark_case, "top_fn", "") or ""),
        ).strip("_")
        return f"{top_fn or 'kernel'}.cpp"

    def parse_generated_output(
        self,
        benchmark_case: Any,
        raw_response: str,
    ) -> tuple[str, str]:
        opening_tags = list(_OUTPUT_OPEN_TAG.finditer(raw_response))
        closing_tags = list(_OUTPUT_CLOSE_TAG.finditer(raw_response))
        if len(opening_tags) != 1 or len(closing_tags) != 1:
            raise ValueError(
                "Expected exactly one OUTPUT_CODE element, found "
                f"{len(opening_tags)} opening and {len(closing_tags)} closing tags"
            )

        opening = opening_tags[0]
        closing = closing_tags[0]
        if opening.end() > closing.start():
            raise ValueError("OUTPUT_CODE closing tag must follow its opening tag")
        opening_name = opening.group(1)
        closing_name = closing.group(1)
        if closing_name is not None and closing_name != opening_name:
            raise ValueError(
                "OUTPUT_CODE opening/closing name mismatch: "
                f"{opening_name!r} != {closing_name!r}"
            )

        if closing_name is None:
            # Preserve the frozen HLS-Eval parser for its canonical prompt form.
            generated = self.extract_code(raw_response)
            if len(generated) != 1:
                raise ValueError(
                    f"Expected exactly one OUTPUT_CODE element, found {len(generated)}"
                )
            filename, code = next(iter(generated.items()))
            if filename != opening_name:
                raise ValueError(
                    "OUTPUT_CODE parser filename mismatch: "
                    f"{opening_name!r} != {filename!r}"
                )
        else:
            # The upstream regex is greedy when the closing tag also has a name.
            # Counts, order, and exact name equality have already been checked.
            filename = opening_name
            code = raw_response[opening.end() : closing.start()]
        if not filename.endswith(".cpp") or filename.endswith("_tb.cpp"):
            raise ValueError(
                f"Generated filename must be a kernel .cpp file: {filename!r}"
            )
        if Path(filename).name != filename or re.search(r"[\\/]", filename):
            raise ValueError(f"Generated filename must not contain a path: {filename!r}")
        if not code.strip():
            raise ValueError("Generated kernel is empty")

        # HLS-Eval's frozen generation prompt uses this literal placeholder. It
        # is transport metadata rather than task knowledge, so normalize it
        # deterministically without penalizing Parse.
        if filename == "kernel_name.cpp":
            filename = self.infer_expected_filename(benchmark_case)
        return filename, code

    def review(
        self,
        benchmark_case: Any,
        raw_response: str,
        round_dir: Path,
        build_name: str,
    ) -> ReviewResult:
        round_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(round_dir / "raw_llm_output.txt", raw_response)
        result = ReviewResult()

        try:
            filename, code = self.parse_generated_output(benchmark_case, raw_response)
            result.pass_parse = True
            result.generated_filename = filename
            result.generated_code = code
        except Exception as exc:
            result.parse_error = str(exc)
            result.failure_stage = FailureStage.PARSE.value
            return result

        design_dir = round_dir / "design_generated"
        if design_dir.exists():
            shutil.rmtree(design_dir)
        design_dir.mkdir(parents=True)
        kernel_path = design_dir / result.generated_filename
        atomic_write_text(kernel_path, result.generated_code or "")

        copied_sources: list[Path] = [kernel_path]
        for source in [benchmark_case.tb_file, *benchmark_case.h_files]:
            destination = design_dir / source.name
            shutil.copy2(source, destination)
            copied_sources.append(destination)

        aux_files: list[Path] = []
        for source in benchmark_case.tb_data_files:
            destination = design_dir / source.name
            shutil.copy2(source, destination)
            aux_files.append(destination)

        build_dir = round_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        common = {
            "build_dir": build_dir,
            "source_files": sorted(copied_sources),
            "aux_files": sorted(aux_files),
            "build_name": build_name,
            "hls_fpga_part": self.fpga_part,
            "hls_clock_period_ns": self.clock_ns,
            "hls_top_function": benchmark_case.top_fn,
            "hls_flow_target": "vivado",
        }
        csim_future = self.pools.csim.submit(
            self.csim_tool.run,
            **common,
            timeout=self.csim_timeout,
        )
        synth_future = self.pools.synth.submit(
            self.synth_tool.run,
            **common,
            timeout=self.synth_timeout,
        )

        try:
            compile_output, run_output = csim_future.result()
            result.compile = _execution_dict(compile_output)
            result.testbench = _execution_dict(run_output)
            result.pass_compile = bool(result.compile and result.compile["return_code"] == 0 and not result.compile["timeout"])
            if getattr(self, "dataset_kind", DATASET_HLS_EVAL) == DATASET_BENCH4HLS:
                oracle = bench4hls_tb_oracle(
                    result.testbench,
                    frozen_epoch=getattr(
                        self,
                        "bench4hls_frozen_epoch",
                        "@2027-01-01 00:00:00",
                    ),
                )
                if result.testbench is not None:
                    result.testbench["oracle"] = oracle
                result.pass_tb = bool(result.pass_compile and oracle["passed"])
            else:
                result.pass_tb = bool(
                    result.pass_compile
                    and result.testbench
                    and result.testbench["return_code"] == 0
                    and not result.testbench["timeout"]
                )
            _append_infrastructure_error(
                result,
                classify_infrastructure_failure(
                    {"compile": result.compile, "testbench": result.testbench}
                ),
            )
        except Exception as exc:
            result.compile = {"return_code": -1, "stdout": "", "stderr": str(exc), "timeout": False}
            _append_infrastructure_error(
                result, f"csim infrastructure exception: {exc!r}"
            )

        try:
            synth_output = synth_future.result()
            result.synthesis = _execution_dict(synth_output)
            result.pass_synth = bool(result.synthesis and result.synthesis["return_code"] == 0 and not result.synthesis["timeout"])
            result.qor = synth_output.data_tool if result.pass_synth else None
            _append_infrastructure_error(
                result,
                classify_infrastructure_failure({"synthesis": result.synthesis}),
            )
        except Exception as exc:
            result.synthesis = {"return_code": -1, "stdout": "", "stderr": str(exc), "timeout": False}
            _append_infrastructure_error(
                result, f"synthesis infrastructure exception: {exc!r}"
            )

        result.pass_tb_and_synth = result.pass_tb and result.pass_synth
        if not result.pass_compile:
            result.failure_stage = FailureStage.COMPILE.value
        elif not result.pass_tb:
            result.failure_stage = FailureStage.TB.value
        elif not result.pass_synth:
            result.failure_stage = FailureStage.SYNTH.value
        else:
            result.failure_stage = FailureStage.SUCCESS.value
        return result
