from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .deadline_process import run_process_file_backed


DEFAULT_LIBFAKETIME = Path(
    "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"
)


def _terminate_isolated_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate a subprocess and every descendant in its isolated group.

    Vitis may let ``csim.exe`` outlive the ``vitis_hls`` group leader.  Killing
    by the leader PID (or taking a one-time psutil descendant snapshot) can
    therefore leave a CPU-bound simulator behind with inherited stdout/stderr
    pipes.  Every Popen guarded by this helper starts a new session, so its PID
    is also the stable process-group ID even after the leader exits.
    """

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(1.0, grace_seconds))


def deterministic_tb_environment(
    base: dict[str, str],
    *,
    libfaketime: Path,
    frozen_epoch: str,
) -> dict[str, str]:
    if not frozen_epoch.startswith("@"):
        raise ValueError(
            "Bench4HLS frozen epoch must use libfaketime absolute '@' syntax"
        )
    env = dict(base)
    old_preload = env.get("LD_PRELOAD", "").strip()
    env["LD_PRELOAD"] = (
        f"{libfaketime}:{old_preload}"
        if old_preload
        else str(libfaketime)
    )
    env["FAKETIME"] = frozen_epoch
    env["FAKETIME_DONT_FAKE_MONOTONIC"] = "1"
    env["FAKETIME_NO_CACHE"] = "1"
    return env


class DeterministicVitisHLSCSimTool:
    """HLS-Eval-compatible C-sim with a frozen wall clock for the TB process.

    Compilation and Vitis project setup use the ordinary environment. Only the
    compiled csim.exe receives libfaketime, so time-seeded Bench4HLS testbenches
    see identical random seeds across modes and retries.
    """

    def __init__(
        self,
        vitis_hls_path: Path,
        frozen_epoch: str,
        libfaketime: Path = DEFAULT_LIBFAKETIME,
    ) -> None:
        from hls_eval.tools import build_vitis_hls_env

        self.vitis_hls_path = vitis_hls_path
        self.frozen_epoch = frozen_epoch
        self.libfaketime = libfaketime.resolve()
        self._build_vitis_hls_env = build_vitis_hls_env
        if not self.libfaketime.is_file():
            raise FileNotFoundError(
                f"Bench4HLS deterministic C-sim requires {self.libfaketime}"
            )
        if not self.frozen_epoch.startswith("@"):
            raise ValueError(
                "Bench4HLS frozen epoch must use libfaketime absolute '@' syntax"
            )

    def run(
        self,
        build_dir: Path,
        source_files: list[Path],
        aux_files: list[Path] = [],
        build_name: str | None = None,
        build_name_prefix: str = "vitis_hls_csim_tool__",
        hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
        hls_clock_period_ns: float = 5,
        hls_top_function: str | None = None,
        hls_flow_target: str = "vivado",
        warn_all: bool = False,
        timeout: float = 60.0 * 2,
    ) -> tuple[Any, Any | None]:
        from hls_eval.tools import (
            ExecutionData,
            ToolDataOutput,
        )

        if build_name is None:
            build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
        else:
            build_name = f"{build_name_prefix}{build_name}"

        unique_build_dir = build_dir / build_name
        if unique_build_dir.exists():
            import shutil

            shutil.rmtree(unique_build_dir)
        unique_build_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        for fp in source_files + aux_files:
            shutil.copy(fp, unique_build_dir)

        tcl_script_fp = unique_build_dir / "run_hls.tcl"
        tcl_script = f"open_project {build_name}__proj\n"
        for fp in source_files:
            if warn_all:
                tcl_script += (
                    'add_files -tb -cflags "-Wall -Wextra '
                    f'-Wno-unused-function" {fp}\n'
                )
            else:
                tcl_script += f"add_files -tb {fp}\n"
        for fp in aux_files:
            tcl_script += f"add_files -tb {fp}\n"
        tcl_script += (
            f"open_solution solution__synth -flow_target {hls_flow_target}\n"
        )
        if hls_top_function is not None:
            tcl_script += f"set_top {hls_top_function}\n"
        tcl_script += f"set_part {hls_fpga_part}\n"
        tcl_script += (
            f"create_clock -period {hls_clock_period_ns} -name clk_default\n"
        )
        tcl_script += "csim_design -setup\nexit\n"
        tcl_script_fp.write_text(tcl_script, encoding="utf-8")

        vitis_hls_bin = self.vitis_hls_path / "bin/vitis_hls"
        vitis_hls_env = self._build_vitis_hls_env(self.vitis_hls_path)

        compile_result = run_process_file_backed(
            [vitis_hls_bin.resolve(), "-f", tcl_script_fp.resolve()],
            cwd=unique_build_dir,
            env=vitis_hls_env,
            timeout=timeout,
            log_prefix=unique_build_dir / "deadline_compile",
        )
        if compile_result.timed_out:
            return (
                ToolDataOutput(
                    data_execution=ExecutionData(
                        return_code=-1,
                        stdout=compile_result.stdout,
                        stderr=compile_result.stderr,
                        t0=compile_result.t0,
                        t1=compile_result.t1,
                        execution_time=timeout,
                        timeout=True,
                    ),
                    data_tool=None,
                ),
                None,
            )

        compile_data = ToolDataOutput(
            data_execution=ExecutionData(
                return_code=compile_result.return_code,
                stdout=compile_result.stdout,
                stderr=compile_result.stderr,
                t0=compile_result.t0,
                t1=compile_result.t1,
                execution_time=compile_result.execution_time,
                timeout=False,
            ),
            data_tool=None,
        )
        if compile_result.return_code != 0:
            return compile_data, None

        csim_exe_fp = (
            unique_build_dir
            / f"{build_name}__proj/solution__synth/csim/build/csim.exe"
        )
        run_env = deterministic_tb_environment(
            vitis_hls_env,
            libfaketime=self.libfaketime,
            frozen_epoch=self.frozen_epoch,
        )

        run_result = run_process_file_backed(
            [csim_exe_fp.resolve()],
            cwd=csim_exe_fp.parent,
            env=run_env,
            timeout=timeout,
            log_prefix=unique_build_dir / "deadline_run",
        )
        if run_result.timed_out:
            return compile_data, ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=-1,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                    t0=run_result.t0,
                    t1=run_result.t1,
                    execution_time=timeout,
                    timeout=True,
                ),
                data_tool=None,
            )

        return compile_data, ToolDataOutput(
            data_execution=ExecutionData(
                return_code=run_result.return_code,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                t0=run_result.t0,
                t1=run_result.t1,
                execution_time=run_result.execution_time,
                timeout=False,
            ),
            data_tool=None,
        )
