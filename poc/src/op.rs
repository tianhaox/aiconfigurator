//! OpSpec — the data-only representation of one operation in a model.
//!
//! Python `Operation` instances are converted (in `engine::build_engine`) into
//! `OpSpec` enum variants.  All execute logic lives below; OpSpec carries the
//! parameters only.

use serde::{Deserialize, Serialize};

/// One operation's spec: kind + execution parameters.
///
/// PoC has only `Gemm`; add more variants as the PoC scope grows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpSpec {
    Gemm {
        name: String,
        scale_factor: f64,
        m: u32,
        n: u32,
        k: u32,
    },
}

impl OpSpec {
    pub fn name(&self) -> &str {
        match self {
            OpSpec::Gemm { name, .. } => name,
        }
    }
}
