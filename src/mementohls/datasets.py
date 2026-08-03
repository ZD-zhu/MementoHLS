from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DATASET_HLS_EVAL = "hls_eval"
DATASET_BENCH4HLS = "bench4hls"
SUPPORTED_DATASETS = (DATASET_HLS_EVAL, DATASET_BENCH4HLS)


BENCH4HLS_ROUND0_SYSTEM = (
    "## Overview\n"
    "You are an expert hardware engineer and C++ developer. Generate a complete "
    "Vitis HLS C++ implementation for the instruction below.\n\n"
    "## Requirements\n"
    "1. Preserve the exact TopModule function name, parameter types, dimensions, "
    "and in-place/output semantics stated by the instruction.\n"
    "2. Produce functionally correct and synthesizable HLS C++ for Vitis HLS.\n"
    "3. Include only headers required by the implementation.\n"
    "4. Do not add main(), a testbench, host I/O, dynamic allocation, recursion, "
    "or placeholders.\n"
    "5. Output one complete design candidate and nothing else.\n\n"
    "## Output Format\n"
    "<OUTPUT_CODE name=\"kernel_name.cpp\">\n"
    "... complete C++ implementation ...\n"
    "</OUTPUT_CODE>\n\n"
    "## Task Instruction\n"
)


@dataclass(frozen=True)
class Bench4HLSCase:
    benchmark_root: Path
    name: str
    prompt_path: Path
    testbench_path: Path

    @property
    def design_dir(self) -> Path:
        return self.benchmark_root

    @property
    def kernel_description_fp(self) -> Path:
        return self.prompt_path

    @property
    def tb_file(self) -> Path:
        return self.testbench_path

    @property
    def h_files(self) -> list[Path]:
        return []

    @property
    def tb_data_files(self) -> list[Path]:
        return []

    @property
    def top_fn(self) -> str:
        return "TopModule"

    @property
    def tags_all(self) -> list[str]:
        return ["Bench4HLS", "instruction_to_hls"]

    @property
    def dataset_id(self) -> str:
        return DATASET_BENCH4HLS

    @property
    def expected_filename(self) -> str:
        return f"{self.name}.cpp"

    @property
    def interface_contract(self) -> str:
        # The public instruction contains the required prototype. Keeping the
        # full instruction is fail-closed when a multiline prototype is unusual.
        return self.prompt_path.read_text(encoding="utf-8")

    @property
    def manifest_files(self) -> tuple[Path, ...]:
        # Golden reference_design files are deliberately absent.
        return (self.prompt_path, self.testbench_path)

    @property
    def manifest_root(self) -> Path:
        return self.benchmark_root


def load_bench4hls_cases(dataset_root: Path) -> list[Bench4HLSCase]:
    root = dataset_root.resolve()
    prompts = root / "prompts"
    testbenches = root / "testbenches"
    if not prompts.is_dir() or not testbenches.is_dir():
        raise ValueError(
            "Bench4HLS dataset root must contain prompts/ and testbenches/: "
            f"{root}"
        )
    cases: list[Bench4HLSCase] = []
    for prompt in sorted(prompts.glob("Prob*_prompt.txt")):
        match = re.fullmatch(r"(Prob\d{3})_prompt\.txt", prompt.name)
        if match is None:
            continue
        name = match.group(1)
        tb = testbenches / f"{name}_tb.cpp"
        if not tb.is_file():
            raise FileNotFoundError(f"Missing Bench4HLS testbench: {tb}")
        if not prompt.read_text(encoding="utf-8").strip():
            raise ValueError(f"Empty Bench4HLS instruction: {prompt}")
        cases.append(Bench4HLSCase(root, name, prompt, tb))
    expected = [f"Prob{index:03d}" for index in range(1, 171)]
    actual = [case.name for case in cases]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "Bench4HLS requires the frozen Prob001..Prob170 set; "
            f"missing={missing}, extra={extra}"
        )
    return cases


def load_cases(
    dataset_kind: str,
    dataset_root: Path,
    hls_eval_case_factory: Callable[..., Any],
    hls_eval_find_dirs: Callable[[Path], list[Path]],
) -> list[Any]:
    if dataset_kind == DATASET_BENCH4HLS:
        return list(load_bench4hls_cases(dataset_root))
    if dataset_kind != DATASET_HLS_EVAL:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")
    paths = hls_eval_find_dirs(dataset_root)
    cases = [hls_eval_case_factory(path, name=path.name) for path in paths]
    cases.sort(key=lambda item: item.name)
    return cases


def build_seed_prompt(
    case: Any,
    dataset_kind: str,
    hls_eval_builder: Callable[[Path, Path, Path], str],
) -> str:
    if dataset_kind == DATASET_BENCH4HLS:
        instruction = case.kernel_description_fp.read_text(encoding="utf-8").strip()
        output_contract = BENCH4HLS_ROUND0_SYSTEM.replace(
            "kernel_name.cpp", case.expected_filename
        )
        return output_contract + instruction + "\n\n## Task Output\n"
    if len(case.h_files) != 1:
        raise ValueError(
            f"HLS-Eval case {case.name} requires exactly one project header"
        )
    return hls_eval_builder(
        case.kernel_description_fp,
        case.tb_file,
        case.h_files[0],
    )


def case_interface_contract(case: Any) -> str:
    explicit = getattr(case, "interface_contract", None)
    if isinstance(explicit, str):
        return explicit
    return "\n\n".join(
        path.read_text(encoding="utf-8")
        for path in case.h_files
    )


def case_header_name(case: Any) -> str:
    return case.h_files[0].name if len(case.h_files) == 1 else ""


def case_expected_filename(case: Any) -> str:
    explicit = getattr(case, "expected_filename", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if len(case.h_files) == 1:
        return f"{case.h_files[0].stem}.cpp"
    return f"{case.name}.cpp"


def case_manifest_files(case: Any) -> tuple[Path, ...]:
    explicit = getattr(case, "manifest_files", None)
    if explicit is not None:
        return tuple(Path(path) for path in explicit)
    return tuple(
        [
            case.kernel_description_fp,
            case.top_file,
            case.design_dir / "hls_eval_config.toml",
            case.tb_file,
            *case.h_files,
            *case.tb_data_files,
        ]
    )


def case_manifest_root(case: Any) -> Path:
    return Path(getattr(case, "manifest_root", case.design_dir)).resolve()
