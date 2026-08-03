# MementoHLS

MementoHLS is a memory-guided, multi-round HLS C/C++ generation and repair system. It connects an OpenAI-compatible language-model endpoint to deterministic parsing, C simulation, testbench validation, and Vitis HLS synthesis, then converts execution evidence into short-term repair state and reusable long-term rules.

This repository is the consolidated code release corresponding to the MementoHLS paper. The code-first layout follows the same style as [KernelMem](https://github.com/0satan0/KernelMem): a top-level entry point, explicit configuration, a memory bank, reusable implementation modules, and operational scripts. Historical documents, experiment outputs, logs, figures, caches, and generated artifacts are intentionally excluded.

## Core workflow

1. `main.py` loads a benchmark, run mode, frozen protocol, model endpoint, and resource limits.
2. `orchestrator.py` generates the initial HLS candidate and coordinates bounded repair rounds.
3. `reviewer.py` evaluates parsing, compilation, testbench execution, and synthesis independently.
4. `memory_facts.py` and `rfl.py` convert trajectory evidence into short-term failure and repair state.
5. `ercl.py` retrieves validated long-term rules from `memorybank/` using stage and contract-aware routing.
6. `semantic_operators.py`, `contract_operators.py`, and `executor.py` turn retrieved guidance into explicit repair actions.
7. `candidate_selection.py`, `trajectory_contracts.py`, and `provenance.py` preserve deterministic selection and auditable trajectories.

## Key features

- Zero-shot, feedback, retrieval-free learning, ERCL, and full MementoHLS execution modes.
- Short-term trajectory memory plus cross-task rule memory.
- Contract-aware repair operators and guarded action execution.
- Clean-room recovery, rule vetoes, causal replay, and candidate selection.
- Independent Parse, Compile, Testbench, Synthesis, and joint TB-and-Synthesis metrics.
- Deterministic C-simulation support, bounded subprocess deadlines, resource gates, and deployment evidence.
- HLS-Eval and Bench4HLS dataset adapters.

## Repository layout

```text
mementohls/
??? main.py                 # Main CLI and experiment orchestration entry point
??? configs/                # HLS-Eval, Bench4HLS, and shared configurations
??? memorybank/             # Active ERCL long-term rule banks
??? scripts/                # Run, deployment, validation, and analysis utilities
??? src/mementohls/         # Core MementoHLS implementation
??? pyproject.toml          # Python package metadata
??? requirements.txt        # Runtime Python dependencies
```

Important implementation modules include:

- `orchestrator.py`: end-to-end generation, review, memory, and repair loop.
- `prompts.py` / `model_input_contract.py`: prompt construction and input contracts.
- `reviewer.py` / `deterministic_csim.py`: stage-isolated HLS evaluation.
- `rfl.py`: short-term repair memory and stagnation control.
- `ercl.py`: long-term rule loading, validation, matching, ranking, and routing.
- `semantic_operators.py` / `contract_operators.py`: repair action definitions.
- `executor.py`: constrained action execution.
- `causal_replay.py`: controlled replay and counterfactual auditing.
- `metrics.py`: exact pass@k aggregation.

## Environment

- Linux with Python 3.10+
- An OpenAI-compatible LLM service
- AMD/Xilinx Vitis HLS for compilation, C simulation, and synthesis
- A local `hls_eval` package or compatible benchmark adapter

Install the Python package and runtime dependencies:

```bash
python -m pip install -e .
```

Vitis HLS and benchmark datasets are external dependencies and are not bundled.

## Usage

Inspect the complete CLI:

```bash
python main.py --help
```

A minimal invocation has the following shape; replace every path and endpoint with values from your environment:

```bash
python main.py \
  --mode mementohls \
  --dataset-kind bench4hls \
  --dataset-root /path/to/Bench4HLS \
  --hls-eval-root /path/to/Hls-Eval \
  --rules memorybank/ercl_primary.yaml \
  --base-url http://localhost:8000/v1 \
  --model your-served-model-name \
  --output-dir /path/to/output
```

Use `configs/` and `scripts/` for the frozen paper-style settings, deployment binding, resource gates, and validation workflow. Generated outputs should remain outside the source tree.

## Scope of this release

The consolidated implementation uses the most recent and most complete v3.19/P5 code line as the canonical version. The DAC_test tree was used to verify the development lineage, and the DAC2027 tree was used to verify the paper-facing implementation. DAC2027 contains no Python module absent from the canonical implementation; its differing modules are earlier revisions of components retained here.
