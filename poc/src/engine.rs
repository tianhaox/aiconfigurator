//! Engine — compiled artifact + execution.
//!
//! Built once from a Python model (op list), called many times.
//! Both the PyO3 binding (Python sweep path) and direct Rust callers
//! (mocker_demo binary) hit the same `run_static`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::op::OpSpec;

/// Which phase to evaluate.  Matches Python's `mode` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticMode {
    /// Context (prefill) phase only.
    Ctx,
    /// Generation (decode) phase only.
    Gen,
    /// Both context and generation.
    Full,
}

impl StaticMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "static_ctx" => Some(StaticMode::Ctx),
            "static_gen" => Some(StaticMode::Gen),
            "static" | "static_full" => Some(StaticMode::Full),
            _ => None,
        }
    }
}

/// Per-op execution result.  Float-only for the PoC.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct OpResult {
    pub latency_ms: f64,
}

/// Compiled model: op lists for each phase + metadata.  Cheap to share
/// across threads (no interior mutability; `Arc<Engine>` for fan-out).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Engine {
    pub context_ops: Vec<OpSpec>,
    pub generation_ops: Vec<OpSpec>,
    /// Free-form metadata for debugging / audit.  Not used by execute.
    pub metadata: HashMap<String, String>,
}

impl Engine {
    /// Build an Engine from already-collected OpSpec lists.  This is what
    /// the PyO3 `build_engine` binding calls.
    pub fn new(context_ops: Vec<OpSpec>, generation_ops: Vec<OpSpec>) -> Self {
        Engine {
            context_ops,
            generation_ops,
            metadata: HashMap::new(),
        }
    }

    /// Serialize to bincode bytes.  The bytes are the architectural
    /// boundary; disk and stdin are just transports.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("serialize: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("deserialize: {e}"))
    }

    pub fn save_bin(&self, path: &Path) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn load_bin(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    // -------------------------------------------------------------------
    // Hot path: same function called by both PyO3 binding and Rust crate
    // direct callers (e.g. mocker_demo).
    // -------------------------------------------------------------------

    /// Execute one static phase (or both).  Returns latency-per-op-name dict.
    ///
    /// - `batch_size`: max concurrent requests
    /// - `seq_len`: effective sequence length for this phase
    /// - `mode`: which phase to compute
    pub fn run_static(
        &self,
        db: &Database,
        batch_size: u32,
        seq_len: u32,
        mode: StaticMode,
    ) -> Result<HashMap<String, OpResult>, String> {
        let mut out: HashMap<String, OpResult> = HashMap::new();
        if matches!(mode, StaticMode::Ctx | StaticMode::Full) {
            execute_phase(&self.context_ops, db, batch_size, seq_len, &mut out)?;
        }
        if matches!(mode, StaticMode::Gen | StaticMode::Full) {
            execute_phase(&self.generation_ops, db, batch_size, seq_len, &mut out)?;
        }
        Ok(out)
    }
}

/// Walk an op list, look each op up in the DB, scale, accumulate.
fn execute_phase(
    ops: &[OpSpec],
    db: &Database,
    batch_size: u32,
    seq_len: u32,
    out: &mut HashMap<String, OpResult>,
) -> Result<(), String> {
    for op in ops {
        let op_name = op.name().to_string();
        let result = execute_op(op, db, batch_size, seq_len)?;
        // Accumulate by name (an op may legitimately appear multiple times in
        // the list with the same name — matches AIC Python convention).
        out.entry(op_name)
            .and_modify(|r| r.latency_ms += result.latency_ms)
            .or_insert(result);
    }
    Ok(())
}

fn execute_op(
    op: &OpSpec,
    db: &Database,
    batch_size: u32,
    seq_len: u32,
) -> Result<OpResult, String> {
    match op {
        OpSpec::Gemm {
            name,
            scale_factor,
            m,
            n,
            k,
        } => {
            // PoC GEMM cost model: parquet has (m, n, k) → latency_ms for a
            // *unit batch×seq*.  Real latency scales linearly with the
            // outer dim x = batch_size * seq_len; multiply, then apply the
            // op's scale_factor (e.g. number of layers).
            let base = db
                .gemm
                .lookup(*m, *n, *k)
                .ok_or_else(|| format!("gemm miss in DB: op={name} (m={m},n={n},k={k})"))?;
            let x = (batch_size as f64) * (seq_len as f64);
            Ok(OpResult {
                latency_ms: base * x * scale_factor,
            })
        }
        OpSpec::Dsa {
            name: _,
            scale_factor,
            num_heads,
            head_dim_qk,
            head_dim_v,
            topk,
            dtype_bytes,
        } => {
            // DSA (DeepSeek Sparse Attention) roofline SoL: each query
            // attends to `topk` selected keys; max(compute, memory) is
            // the bottleneck wall — whichever side dominates sets latency.
            let queries = (batch_size as f64) * (seq_len as f64);
            let nh = *num_heads as f64;
            let hk = *head_dim_qk as f64;
            let hv = *head_dim_v as f64;
            let tk = *topk as f64;
            let db_bytes = *dtype_bytes as f64;

            let flops_per_query = nh * 2.0 * tk * (hk + hv);
            let bytes_per_query = nh * tk * (hk + hv) * db_bytes;
            let total_flops = queries * flops_per_query;
            let total_bytes = queries * bytes_per_query;

            let compute_ms = total_flops / (db.gpu.peak_tflops_bf16 * 1e12) * 1e3;
            let mem_ms = total_bytes / (db.gpu.hbm_bw_gbps * 1e9) * 1e3;
            let sol_ms = compute_ms.max(mem_ms);
            Ok(OpResult {
                latency_ms: sol_ms * scale_factor,
            })
        }
    }
}

// Correctness testing happens end-to-end via the Python parity test
// (`tests/test_e2e.py`).  Rust-side unit tests can be added later when
// the surface grows beyond a single op kind.
