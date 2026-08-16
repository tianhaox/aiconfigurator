# SPDX-License-Identifier: Apache-2.0
"""End-to-end PoC tests.

What we verify:
    1. PyO3 binding imports cleanly.
    2. Python model → Rust Engine round-trip works (build_engine).
    3. Engine.run_static returns the SAME per-op latencies as the pure
       Python reference implementation.
    4. The Engine can be serialized (.bin) and reloaded.
    5. DbHandle.from_dict matches DbHandle.load(parquet).

Anything in here failing means the PoC's core promise (Python +
Rust bit-identical) is broken.
"""

from __future__ import annotations

import math
from pathlib import Path

import aic_step
import pytest
from aic_poc import MockLLMModel, run_static_reference

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
PARQUET = DATA / "gemm_perf.parquet"


@pytest.fixture(scope="module")
def db():
    """Load the test parquet through the Rust DbHandle."""
    if not PARQUET.exists():
        # Build it on the fly if missing — keeps the test self-sufficient.
        from data.build_gemm_parquet import main as build

        build()
    return aic_step.DbHandle.load(str(PARQUET))


@pytest.fixture(scope="module")
def py_db():
    """Pure-Python dict mirror of the parquet for the reference implementation."""
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET)
    rows = table.to_pylist()
    return {(r["m"], r["n"], r["k"]): r["latency_ms"] for r in rows}


@pytest.fixture(scope="module")
def model():
    return MockLLMModel(hidden=4096, n_layers=32)


@pytest.fixture(scope="module")
def engine(model):
    """Build the Rust Engine from the Python model.  Stamp db_id so the
    compat check sees matched engine/db pairs in downstream tests."""
    return aic_step.build_engine(model, metadata={"db_id": model.DB_ID})


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_imports():
    assert hasattr(aic_step, "Engine")
    assert hasattr(aic_step, "DbHandle")
    assert hasattr(aic_step, "build_engine")
    assert hasattr(aic_step, "load_engine")


def test_engine_basic_props(engine):
    # ctx: qkv_proj, out_proj, dsa_ctx  (3) + gen: decode_qkv, dsa_gen (2)
    assert engine.op_count() == 5


def test_dbhandle_parquet_load(db):
    assert db.gemm_table_size() >= 2  # qkv_proj + out_proj shapes


# ---------------------------------------------------------------------------
# Parity (the main event)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["static_ctx", "static_gen", "static"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (1, 1024),
        (8, 2048),
        (32, 4096),
    ],
)
def test_rust_matches_python_reference(engine, db, py_db, model, mode, batch, seq):
    """Python ref vs Rust Engine — every op's latency must match exactly."""
    rust_result = engine.run_static(db, batch, seq, mode)
    py_result = run_static_reference(model, py_db, batch, seq, mode)

    assert set(rust_result.keys()) == set(py_result.keys()), (
        f"op-name mismatch: rust={set(rust_result)}, py={set(py_result)}"
    )
    for name, py_val in py_result.items():
        rust_val = rust_result[name]
        # Floats from Rust f64 ↔ Python float; allow tiny rounding noise
        # but expect bit-identical for these clean numbers.
        assert math.isclose(rust_val, py_val, rel_tol=1e-12, abs_tol=1e-12), (
            f"op={name!r} mode={mode!r} batch={batch} seq={seq}: rust={rust_val!r} vs py={py_val!r}"
        )


# ---------------------------------------------------------------------------
# Artifact round-trip
# ---------------------------------------------------------------------------


def test_engine_save_load_round_trip(engine, db, py_db, model, tmp_path):
    """Engine.save_bin → load_engine produces an Engine that runs identically."""
    bin_path = tmp_path / "compiled.bin"
    engine.save_bin(str(bin_path))
    assert bin_path.exists()

    reloaded = aic_step.load_engine(str(bin_path))
    assert reloaded.op_count() == engine.op_count()

    rust_a = engine.run_static(db, 8, 2048, "static")
    rust_b = reloaded.run_static(db, 8, 2048, "static")
    assert rust_a == rust_b

    # And both still match the pure-Python reference.
    py = run_static_reference(model, py_db, 8, 2048, "static")
    assert set(rust_b.keys()) == set(py.keys())
    for name, py_val in py.items():
        assert math.isclose(rust_b[name], py_val, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Bytes-boundary round-trip
# ---------------------------------------------------------------------------


def test_engine_bytes_round_trip(engine, db):
    """Engine.to_bytes → engine_from_bytes runs identically (in-memory transport)."""
    blob = engine.to_bytes()
    assert isinstance(blob, bytes) and len(blob) > 0
    e2 = aic_step.engine_from_bytes(blob)
    a = engine.run_static(db, 8, 2048, "static")
    b = e2.run_static(db, 8, 2048, "static")
    assert a == b


# ---------------------------------------------------------------------------
# DbHandle from-dict shortcut
# ---------------------------------------------------------------------------


def test_dbhandle_from_dict_matches_parquet_load(engine, db, py_db, model):
    db2 = aic_step.DbHandle.from_dict(py_db, metadata={"db_id": model.DB_ID})
    out_a = engine.run_static(db, 4, 1024, "static")
    out_b = engine.run_static(db2, 4, 1024, "static")
    assert out_a == out_b


# ---------------------------------------------------------------------------
# Engine ↔ DB compat metadata
# ---------------------------------------------------------------------------


def test_db_carries_parquet_metadata(db, model):
    """Parquet kv-metadata flows through into DbHandle.metadata."""
    assert db.metadata("db_id") == model.DB_ID


def test_engine_carries_build_metadata(engine, model):
    assert engine.metadata("db_id") == model.DB_ID


def test_check_db_compat_passes_on_match(engine, db):
    engine.check_db_compat(db)  # should not raise


def test_check_db_compat_fails_on_mismatch(model, py_db):
    e = aic_step.build_engine(model, metadata={"db_id": "wrong_gpu"})
    db_wrong = aic_step.DbHandle.from_dict(py_db, metadata={"db_id": model.DB_ID})
    with pytest.raises(RuntimeError, match="db_id mismatch"):
        e.check_db_compat(db_wrong)


def test_check_db_compat_passes_when_engine_unstamped(model, py_db):
    """Engine with no db_id is treated as 'don't care' — backward compat."""
    e = aic_step.build_engine(model)  # no metadata
    db_any = aic_step.DbHandle.from_dict(py_db, metadata={"db_id": "anything"})
    e.check_db_compat(db_any)  # should not raise
