//! `mocker_demo` — pretend we're Dynamo Mocker calling AIC's Rust engine.
//!
//! Primary mode: read the Engine artifact from **stdin** as bincode bytes
//! (the bytes are the architectural boundary; stdin / file / shared memory
//! are just transports). Alternate mode: pass an explicit file path on the
//! CLI with `--file=<path>` (or as the first positional arg if the file
//! exists on disk).
//!
//! In either case the perf DB is loaded from a parquet file (positional
//! arg, default `data/gemm_perf.parquet`).
//!
//! Then evaluates a few (batch, seq_len) points in parallel via rayon,
//! showing that:
//!  - the Rust core is callable without any Python
//!  - the same `run_static` is the hot path
//!  - multi-thread fan-out is trivial

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use aic_step::{Database, Engine, StaticMode};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Resolve engine source: --file=<p>, or a positional path that exists
    // on disk, or fall back to stdin bytes.
    let mut engine_file: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    for a in &args {
        if let Some(rest) = a.strip_prefix("--file=") {
            engine_file = Some(rest.to_string());
        } else {
            positional.push(a.clone());
        }
    }
    if engine_file.is_none() {
        if let Some(first) = positional.first() {
            if Path::new(first).exists() {
                engine_file = Some(first.clone());
                positional.remove(0);
            }
        }
    }

    let parquet_path = positional
        .into_iter()
        .next()
        .unwrap_or_else(|| "data/gemm_perf.parquet".to_string());

    let engine = if let Some(p) = engine_file {
        println!("[mocker_demo] loading engine from file: {p}");
        Arc::new(Engine::load_bin(Path::new(&p))?)
    } else {
        println!("[mocker_demo] reading engine bytes from stdin");
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| format!("stdin read: {e}"))?;
        Arc::new(Engine::from_bytes(&buf)?)
    };
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
        let result = engine.run_static(&db, *batch, *seq, StaticMode::Full)?;
        let total_ms: f64 = result.values().map(|r| r.latency_ms).sum();
        println!("  batch={batch:4} seq={seq:5} → total_latency_ms={total_ms:10.4}");
    }
    let seq_elapsed = t0.elapsed();

    println!("[mocker_demo] running same points (rayon, parallel):");
    let t0 = Instant::now();
    let totals: Vec<f64> = points
        .par_iter()
        .map(|(batch, seq)| -> Result<f64, String> {
            let result = engine.run_static(&db, *batch, *seq, StaticMode::Full)?;
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
