//! PyO3 bindings for the Engine + DbHandle.
//!
//! Compiled only when the `python` feature is on (it is, when maturin builds
//! the cdylib).  Binaries like `mocker_demo` build with
//! ``--no-default-features`` and skip this module entirely.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::db::{Database, GemmTable, GpuSpec};
use crate::engine::{Engine, StaticMode};
use crate::op::OpSpec;

// ---------------------------------------------------------------------------
// PyEngine
// ---------------------------------------------------------------------------

#[pyclass(name = "Engine", module = "aic_step._native")]
pub struct PyEngine {
    inner: Arc<Engine>,
}

#[pymethods]
impl PyEngine {
    fn op_count(&self) -> usize {
        self.inner.context_ops.len() + self.inner.generation_ops.len()
    }

    #[pyo3(signature = (db, batch_size, seq_len, mode))]
    fn run_static<'py>(
        &self,
        py: Python<'py>,
        db: &PyDbHandle,
        batch_size: u32,
        seq_len: u32,
        mode: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mode = StaticMode::from_str(mode).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown mode {mode:?}; expected 'static_ctx' | 'static_gen' | 'static'"
            ))
        })?;
        let inner = Arc::clone(&self.inner);
        let db_arc = Arc::clone(&db.inner);

        let result = py
            .allow_threads(move || inner.run_static(&db_arc, batch_size, seq_len, mode))
            .map_err(PyRuntimeError::new_err)?;

        let out = PyDict::new_bound(py);
        for (name, op_result) in result {
            out.set_item(name, op_result.latency_ms)?;
        }
        Ok(out)
    }

    fn save_bin(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_bin(std::path::Path::new(path))
            .map_err(PyRuntimeError::new_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    fn metadata(&self, key: &str) -> Option<String> {
        self.inner.metadata.get(key).cloned()
    }

    fn check_db_compat(&self, db: &PyDbHandle) -> PyResult<()> {
        self.inner
            .check_db_compat(&db.inner)
            .map_err(PyRuntimeError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Engine ctx_ops={} gen_ops={}>",
            self.inner.context_ops.len(),
            self.inner.generation_ops.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyDbHandle
// ---------------------------------------------------------------------------

#[pyclass(name = "DbHandle", module = "aic_step._native")]
pub struct PyDbHandle {
    inner: Arc<Database>,
}

#[pymethods]
impl PyDbHandle {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let pb = PathBuf::from(path);
        let inner = Database::load_gemm_parquet(&pb).map_err(PyRuntimeError::new_err)?;
        Ok(PyDbHandle { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (d, metadata=None))]
    fn from_dict(
        d: &Bound<'_, PyDict>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut table = GemmTable::default();
        for (key, value) in d.iter() {
            let (m, n, k): (u32, u32, u32) = key.extract()?;
            let lat: f64 = value.extract()?;
            table.insert(m, n, k, lat);
        }
        let md = extract_metadata(metadata)?;
        Ok(PyDbHandle {
            inner: Arc::new(Database {
                gemm: table,
                gpu: GpuSpec::default(),
                metadata: md,
            }),
        })
    }

    fn gemm_table_size(&self) -> usize {
        self.inner.gemm.len()
    }

    fn metadata(&self, key: &str) -> Option<String> {
        self.inner.metadata.get(key).cloned()
    }

    fn __repr__(&self) -> String {
        format!("<DbHandle gemm_rows={}>", self.inner.gemm.len())
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (model, metadata=None))]
fn build_engine(
    model: &Bound<'_, PyAny>,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyEngine> {
    let context_ops = extract_op_list(model.getattr("context_ops")?)?;
    let generation_ops = extract_op_list(model.getattr("generation_ops")?)?;
    let mut engine = Engine::new(context_ops, generation_ops);
    engine.metadata = extract_metadata(metadata)?;
    Ok(PyEngine {
        inner: Arc::new(engine),
    })
}

fn extract_metadata(d: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, String>> {
    let Some(d) = d else {
        return Ok(HashMap::new());
    };
    let mut out = HashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val: String = v.extract()?;
        out.insert(key, val);
    }
    Ok(out)
}

fn extract_op_list(seq: Bound<'_, PyAny>) -> PyResult<Vec<OpSpec>> {
    let list: Bound<PyList> = seq.downcast_into().map_err(|_| {
        PyValueError::new_err("model.context_ops / generation_ops must be a list")
    })?;
    list.iter().map(|item| extract_op_spec(&item)).collect()
}

fn extract_op_spec(op: &Bound<'_, PyAny>) -> PyResult<OpSpec> {
    let kind: String = op.getattr("op_kind")?.extract()?;
    let name: String = op.getattr("_name")?.extract()?;
    let scale_factor: f64 = op.getattr("_scale_factor")?.extract()?;
    match kind.as_str() {
        "gemm" => {
            let m: u32 = op.getattr("_m")?.extract()?;
            let n: u32 = op.getattr("_n")?.extract()?;
            let k: u32 = op.getattr("_k")?.extract()?;
            Ok(OpSpec::Gemm {
                name,
                scale_factor,
                m,
                n,
                k,
            })
        }
        "dsa" => {
            let num_heads: u32 = op.getattr("_num_heads")?.extract()?;
            let head_dim_qk: u32 = op.getattr("_head_dim_qk")?.extract()?;
            let head_dim_v: u32 = op.getattr("_head_dim_v")?.extract()?;
            let topk: u32 = op.getattr("_topk")?.extract()?;
            let dtype_bytes: u8 = op.getattr("_dtype_bytes")?.extract()?;
            Ok(OpSpec::Dsa {
                name,
                scale_factor,
                num_heads,
                head_dim_qk,
                head_dim_v,
                topk,
                dtype_bytes,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown op_kind {other:?}; PoC supports 'gemm' | 'dsa'"
        ))),
    }
}

#[pyfunction]
fn load_engine(path: &str) -> PyResult<PyEngine> {
    let engine = Engine::load_bin(std::path::Path::new(path)).map_err(PyRuntimeError::new_err)?;
    Ok(PyEngine {
        inner: Arc::new(engine),
    })
}

#[pyfunction]
fn engine_from_bytes(bytes: &[u8]) -> PyResult<PyEngine> {
    let engine = Engine::from_bytes(bytes).map_err(PyRuntimeError::new_err)?;
    Ok(PyEngine {
        inner: Arc::new(engine),
    })
}

// ---------------------------------------------------------------------------
// Module entry
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyDbHandle>()?;
    m.add_function(wrap_pyfunction!(build_engine, m)?)?;
    m.add_function(wrap_pyfunction!(load_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_from_bytes, m)?)?;
    Ok(())
}
