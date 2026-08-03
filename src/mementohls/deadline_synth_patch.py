from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .deadline_process import run_process_file_backed


def _deadline_synth_run(
    self: Any,
    build_dir: Path,
    source_files: list[Path],
    aux_files: list[Path] = [],
    build_name: str | None = None,
    build_name_prefix: str = "vitis_hls_synth_tool__",
    hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
    hls_clock_period_ns: float = 5,
    hls_top_function: str | None = None,
    hls_flow_target: str = "vivado",
    hls_unsafe_math: bool = True,
    timeout: float = 60.0 * 6,
) -> Any:
    """Contract-equivalent Vitis synthesis with file-backed deadline I/O."""

    from hls_eval.tools import (
        ExecutionData,
        ToolDataOutput,
        build_vitis_hls_env,
    )
    from hls_eval.vhls_report import DesignHLSSynthData

    if build_name is None:
        build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
    else:
        build_name = f"{build_name_prefix}{build_name}"

    unique_build_dir = build_dir / build_name
    if unique_build_dir.exists():
        shutil.rmtree(unique_build_dir)
    unique_build_dir.mkdir(parents=True, exist_ok=True)

    for source in source_files + aux_files:
        shutil.copy(source, unique_build_dir)

    tcl_script_fp = unique_build_dir / "run_hls.tcl"
    tcl_script = f"open_project {build_name}__proj\n"
    for source in source_files:
        tcl_script += f"add_files {source}\n"
    tcl_script += (
        f"open_solution solution__synth -flow_target {hls_flow_target}\n"
    )
    if hls_top_function is not None:
        tcl_script += f"set_top {hls_top_function}\n"
    tcl_script += f"set_part {hls_fpga_part}\n"
    tcl_script += (
        f"create_clock -period {hls_clock_period_ns} -name clk_default\n"
    )
    if hls_unsafe_math:
        tcl_script += "config_compile -unsafe_math_optimizations\n"
    tcl_script += "csynth_design\nexit\n"
    tcl_script_fp.write_text(tcl_script, encoding="utf-8")

    vitis_hls_bin = self.vitis_hls_path / "bin/vitis_hls"
    result = run_process_file_backed(
        [vitis_hls_bin.resolve(), "-f", tcl_script_fp.resolve()],
        cwd=unique_build_dir,
        env=build_vitis_hls_env(self.vitis_hls_path),
        timeout=timeout,
        log_prefix=unique_build_dir / "deadline_synth",
    )
    execution = ExecutionData(
        return_code=-1 if result.timed_out else result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
        t0=result.t0,
        t1=result.t1,
        execution_time=timeout if result.timed_out else result.execution_time,
        timeout=result.timed_out,
    )
    if result.timed_out or result.return_code != 0:
        return ToolDataOutput(data_execution=execution, data_tool=None)

    solution_dir = (
        unique_build_dir / f"{build_name}__proj/solution__synth"
    )
    synthesis_data = DesignHLSSynthData.parse_from_synth_report_file(
        solution_dir / "syn" / "report" / "csynth.xml"
    )
    return ToolDataOutput(
        data_execution=execution,
        data_tool=synthesis_data.to_dict(),
    )


def install_deadline_synth_patch() -> None:
    """Patch only VitisHLSSynthTool.run; keep its class and public contract."""

    from hls_eval import tools

    current = tools.VitisHLSSynthTool.run
    if getattr(current, "_mementohls_file_backed_deadline", False):
        return
    _deadline_synth_run._mementohls_file_backed_deadline = True  # type: ignore[attr-defined]
    tools.VitisHLSSynthTool.run = _deadline_synth_run
