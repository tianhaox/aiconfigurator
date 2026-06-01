//! PyO3 bindings for the Engine + DbHandle.
//!
//! Compiled only when the `python` feature is on (it is, when maturin builds
//! the cdylib).  Binaries like `mocker_demo` build with
//! ``--no-default-features`` and skip this module entirely.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::db::{Database, GemmTable};
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
            .allow_threads(move || inner.run_static_internal(&db_arc, batch_size, seq_len, mode))
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
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut table = GemmTable::default();
        for (key, value) in d.iter() {
            let (m, n, k): (u32, u32, u32) = key.extract()?;
            let lat: f64 = value.extract()?;
            table.insert(m, n, k, lat);
        }
        Ok(PyDbHandle {
            inner: Arc::new(Database { gemm: table }),
        })
    }

    fn gemm_table_size(&self) -> usize {
        self.inner.gemm.len()
    }

    fn __repr__(&self) -> String {
        format!("<DbHandle gemm_rows={}>", self.inner.gemm.len())
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

#[pyfunction]
fn build_engine(model: &Bound<'_, PyAny>) -> PyResult<PyEngine> {
    let context_ops = extract_op_list(model.getattr("context_ops")?)?;
    let generation_ops = extract_op_list(model.getattr("generation_ops")?)?;
    let engine = Engine::new(context_ops, generation_ops);
    Ok(PyEngine {
        inner: Arc::new(engine),
    })
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
        other => Err(PyValueError::new_err(format!(
            "unknown op_kind {other:?}; PoC only supports 'gemm'"
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

// ---------------------------------------------------------------------------
// Module entry
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyDbHandle>()?;
    m.add_function(wrap_pyfunction!(build_engine, m)?)?;
    m.add_function(wrap_pyfunction!(load_engine, m)?)?;
    Ok(())
}
