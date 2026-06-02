//! OpSpec — the data-only representation of one operation in a model.
//!
//! Python `Operation` instances are converted (in `engine::build_engine`) into
//! `OpSpec` enum variants.  All execute logic lives below; OpSpec carries the
//! parameters only.

use serde::{Deserialize, Serialize};

/// One operation's spec: kind + execution parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpSpec {
    Gemm {
        name: String,
        scale_factor: f64,
        m: u32,
        n: u32,
        k: u32,
    },
    Dsa {
        name: String,
        scale_factor: f64,
        num_heads: u32,
        head_dim_qk: u32,
        head_dim_v: u32,
        topk: u32,
        dtype_bytes: u8,
    },
}

impl OpSpec {
    pub fn name(&self) -> &str {
        match self {
            OpSpec::Gemm { name, .. } => name,
            OpSpec::Dsa { name, .. } => name,
        }
    }
}
