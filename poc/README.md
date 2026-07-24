# Rust PoC — Greenfield Python + Rust Split

A minimal, self-contained proof-of-concept showing how AIC's perf engine
could be split across Python (model building) and Rust (execution), with
**zero changes to the existing `src/aiconfigurator/` codebase**.

This directory is throwaway: the goal is to validate the architecture
end-to-end before deciding whether to invest in a real refactor.

## What it demonstrates

| # | Claim | How to verify |
|---|---|---|
| 1 | PyO3 binding builds and imports cleanly | `maturin develop && python -c "import aic_step; print(aic_step.Engine)"` |
| 2 | Python model → Rust Engine round-trip works | `pytest tests/test_e2e.py::test_engine_basic_props` |
| 3 | Rust execution is bit-identical to a Python reference | `pytest tests/test_e2e.py::test_rust_matches_python_reference` |
| 4 | Engine can be persisted (`.bin`) and reloaded | `pytest tests/test_e2e.py::test_engine_save_load_round_trip` |
| 5 | External Rust caller can run the same engine | `cargo run --bin mocker_demo` |
| 6 | Multi-threading (rayon) works without GIL contention | `mocker_demo` prints sequential vs parallel wallclock |

## Architecture in one picture

```
Python                                        Rust
─────────────────────────                     ─────────────────────────
MockLLMModel
  .context_ops = [MockGemmOp, MockGemmOp]
  .generation_ops = [MockGemmOp]
        │
        │  aic_step.build_engine(model)
        ▼
                                              Engine
                                                .context_ops:    Vec<OpSpec>
                                                .generation_ops: Vec<OpSpec>
        ▲                                       │
        │ PyEngine (PyO3 wrapper)                │
        │                                       ▼
                                              run_static_internal(db, batch, seq, mode)
                                                  for op in op_list:
                                                      db.gemm.lookup(m, n, k) * x * scale
                                                  → HashMap<name, f64>
        ▲                                       ▲
        │ Python callers (this dir's tests)     │ Rust callers (mocker_demo)
```

The **same `run_static_internal`** function serves both Python sweep
(via PyO3) and external Rust tools.  The PyO3 wrapper releases the GIL
during execution, so Python-side multithreading also works.

## Layout

```
poc/
├── Cargo.toml                   # Rust crate (cdylib + rlib + binary)
├── pyproject.toml               # Maturin / Python package metadata
├── src/
│   ├── lib.rs                   # PyO3 module + #[pyclass] wrappers
│   ├── op.rs                    # OpSpec enum (only Gemm in PoC)
│   ├── db.rs                    # Database + parquet loader
│   ├── engine.rs                # Engine + run_static_internal (hot path)
│   └── bin/mocker_demo.rs       # External Rust caller demo
├── python/
│   ├── aic_step/                # Python facade re-exporting Rust symbols
│   │   └── __init__.py
│   └── aic_poc/                 # Pure-Python helpers (mock model + ref impl)
│       ├── __init__.py
│       ├── mock_model.py
│       └── reference.py
├── data/
│   ├── build_gemm_parquet.py    # Generate test perf data
│   └── gemm_perf.parquet        # (generated)
└── tests/
    └── test_e2e.py              # Python + Rust parity tests
```

## Building & running

### One-time setup

```bash
brew install rust                # or rustup
pip install maturin pyarrow pytest
```

### Build the PyO3 extension into your current Python env

```bash
cd poc/
maturin develop --release
```

This compiles `src/lib.rs` into a shared library and installs it as
`aic_step._native` in your active Python.  After this:

```python
>>> import aic_step
>>> aic_step.Engine
<class 'aic_step._native.Engine'>
```

### Generate test perf data

```bash
python data/build_gemm_parquet.py
```

Produces `data/gemm_perf.parquet` (~4 rows).

### Run the Python parity tests

```bash
pytest tests/test_e2e.py -v
```

Expected: all tests pass.  These tests are the PoC's main success criterion.

### Run the external-Rust-caller demo

First, produce a `compiled.bin` from Python (the test does this in
``tmp_path``; for the demo we'll write it explicitly):

```bash
python -c "
from aic_poc import MockLLMModel
from aic_step import build_engine
m = MockLLMModel(hidden=4096, n_layers=32)
e = build_engine(m)
e.save_bin('compiled.bin')
print('wrote compiled.bin')
"
```

Then build and run the Rust binary:

```bash
cargo build --release --bin mocker_demo
./target/release/mocker_demo compiled.bin data/gemm_perf.parquet
```

You should see sequential and rayon-parallel timing for the same set of
points, plus per-point latency.  This confirms:

- The Rust crate is usable without any Python interpreter loaded.
- The bincode artifact written by Python loads correctly in Rust.
- Multi-threading via `rayon::par_iter` works on the engine without
  changing call sites.

## Scope deliberately omitted

This PoC stops at proving the architecture.  Things deliberately **not**
attempted:

- More than one op kind (only `Gemm`).
- More than one model class (only `MockLLMModel`).
- DB interpolation / extrapolation (exact-match only).
- Quant-mode-aware DB lookups.
- Production CI / multi-platform wheels.
- Integration with the existing `src/aiconfigurator/` SDK.
- Pareto / sweep orchestration (would live in Python on top of this engine).

## Findings → decisions

After running through the verification checklist above, the writeup at
the top of the next PR (whichever-it-becomes) should summarize:

1. Did the architecture hold?  Any sharp edges in PyO3 / maturin
   workflow?
2. Real wallclock numbers for the rayon demo — does parallelism
   actually buy what it advertises?
3. What would change if we expand to all ~10 op kinds?  Maintenance
   cost estimate for the full split.
