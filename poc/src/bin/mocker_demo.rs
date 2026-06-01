//! `mocker_demo` — pretend we're Dynamo Mocker calling AIC's Rust engine.
//!
//! Loads two artifacts produced by the Python build step:
//!  - `compiled.bin`   — bincode-serialized `Engine`
//!  - `gemm.parquet`   — perf data
//!
//! Then evaluates a few (batch, seq_len) points in parallel via rayon,
//! showing that:
//!  - the Rust core is callable without any Python
//!  - the same `run_static_internal` is the hot path
//!  - multi-thread fan-out is trivial

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use aic_step::{Database, Engine, StaticMode};

fn main() -> Result<(), String> {
    let engine_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "compiled.bin".to_string());
    let parquet_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data/gemm_perf.parquet".to_string());

    println!("[mocker_demo] loading engine: {engine_path}");
    let engine = Arc::new(Engine::load_bin(Path::new(&engine_path))?);
    println!(
        "[mocker_demo]   ctx_ops={}, gen_ops={}",
        engine.context_ops.len(),
        engine.generation_ops.len()
    );

    println!("[mocker_demo] loading db: {parquet_path}");
    let db = Database::load_gemm_parquet(Path::new(&parquet_path))?;
    println!("[mocker_demo]   gemm_rows={}", db.gemm.len());

    // A handful of independent points; pretend each one comes from a
    // scheduler telemetry sample.
    let points: Vec<(u32, u32)> = vec![
        (1, 1024),
        (2, 1024),
        (4, 1024),
        (8, 1024),
        (16, 2048),
        (32, 2048),
        (64, 4096),
        (128, 4096),
    ];

    println!("[mocker_demo] running {} points (sequential):", points.len());
    let t0 = Instant::now();
    for (batch, seq) in &points {
        let result = engine.run_static_internal(&db, *batch, *seq, StaticMode::Full)?;
        let total_ms: f64 = result.values().map(|r| r.latency_ms).sum();
        println!("  batch={batch:4} seq={seq:5} → total_latency_ms={total_ms:10.4}");
    }
    let seq_elapsed = t0.elapsed();

    println!("[mocker_demo] running same points (rayon, parallel):");
    let t0 = Instant::now();
    let totals: Vec<f64> = points
        .par_iter()
        .map(|(batch, seq)| -> Result<f64, String> {
            let result = engine.run_static_internal(&db, *batch, *seq, StaticMode::Full)?;
            Ok(result.values().map(|r| r.latency_ms).sum())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let par_elapsed = t0.elapsed();
    for ((batch, seq), total) in points.iter().zip(totals.iter()) {
        println!("  batch={batch:4} seq={seq:5} → total_latency_ms={total:10.4}");
    }

    println!("[mocker_demo]");
    println!("[mocker_demo] sequential wallclock : {:?}", seq_elapsed);
    println!("[mocker_demo] parallel   wallclock : {:?}", par_elapsed);
    println!("[mocker_demo] (point count = {})", points.len());

    Ok(())
}
