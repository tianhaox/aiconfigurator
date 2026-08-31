//! DbHandle — perf database loaded into Rust-native indexed tables.
//!
//! For the PoC we support exactly one table: GEMM latency keyed by
//! (m, n, k) → latency_ms.  Real data would interpolate over a richer key
//! space; PoC keeps it as an exact-match HashMap.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, UInt32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

/// Internal representation of the GEMM table.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GemmTable {
    /// Exact-match lookup: (m, n, k) → latency_ms.
    by_shape: HashMap<(u32, u32, u32), f64>,
}

impl GemmTable {
    pub fn lookup(&self, m: u32, n: u32, k: u32) -> Option<f64> {
        self.by_shape.get(&(m, n, k)).copied()
    }

    pub fn len(&self) -> usize {
        self.by_shape.len()
    }

    /// Insert a (m, n, k) → latency_ms entry.  Useful for tests and for
    /// constructing small fixtures without going through parquet.
    pub fn insert(&mut self, m: u32, n: u32, k: u32, latency_ms: f64) {
        self.by_shape.insert((m, n, k), latency_ms);
    }
}

/// GPU peak specs used by analytic SoL (roofline) cost models.
/// Defaults are H100 SXM5.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuSpec {
    pub peak_tflops_bf16: f64,
    pub hbm_bw_gbps: f64,
}

impl Default for GpuSpec {
    fn default() -> Self {
        GpuSpec {
            peak_tflops_bf16: 990.0,
            hbm_bw_gbps: 3350.0,
        }
    }
}

/// Top-level database handle.  Owns per-op tables.  Shared via `Arc` for
/// cheap clones across threads / sweep points.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Database {
    pub gemm: GemmTable,
    pub gpu: GpuSpec,
    /// Free-form identity stamps (e.g. `db_id="h200_trtllm_1.0.0"`).
    /// Used by `Engine::check_db_compat` to fail fast on engine/db mismatches.
    pub metadata: HashMap<String, String>,
    // future: pub context_attention: AttnTable, etc.
}

impl Database {
    /// Load a parquet file into the GEMM table.  Schema: columns `m`, `n`,
    /// `k` (UInt32), `latency_ms` (Float64).  Any parquet kv-metadata is
    /// copied into `Database.metadata`.
    pub fn load_gemm_parquet(path: &Path) -> Result<Arc<Self>, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("parquet open: {e}"))?;

        // Pull kv-metadata before the builder is consumed by build().
        let metadata: HashMap<String, String> = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .map(|kvs| {
                kvs.iter()
                    .filter_map(|kv| kv.value.as_ref().map(|v| (kv.key.clone(), v.clone())))
                    .collect()
            })
            .unwrap_or_default();

        let mut reader = builder.build().map_err(|e| format!("parquet build: {e}"))?;

        let mut table = GemmTable::default();
        while let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| format!("batch: {e}"))?;
            let m = column_u32(&batch, "m")?;
            let n = column_u32(&batch, "n")?;
            let k = column_u32(&batch, "k")?;
            let lat = column_f64(&batch, "latency_ms")?;
            for i in 0..batch.num_rows() {
                table.by_shape.insert(
                    (m.value(i), n.value(i), k.value(i)),
                    lat.value(i),
                );
            }
        }
        Ok(Arc::new(Database {
            gemm: table,
            gpu: GpuSpec::default(),
            metadata,
        }))
    }

    /// Construct an empty Database (useful for testing without a parquet file).
    pub fn empty() -> Arc<Self> {
        Arc::new(Database::default())
    }
}

fn column_u32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<&'a UInt32Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column '{name}'"))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| format!("column '{name}' is not UInt32"))
}

fn column_f64<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<&'a Float64Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column '{name}'"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("column '{name}' is not Float64"))
}
