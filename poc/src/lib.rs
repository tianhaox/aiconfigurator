//! `aic_step` — Greenfield PoC core.
//!
//! Two consumers, one core:
//!
//! 1. **Python sweep** (when built with `python` feature): the PyO3 binding
//!    in `py.rs` exposes Engine + DbHandle + build_engine to Python.
//! 2. **External Rust callers** (e.g. `bin/mocker_demo.rs`): use this crate
//!    as a normal Rust library (build with ``--no-default-features``); the
//!    Engine + Database types below are the entire surface.

mod db;
mod engine;
mod op;

pub use crate::db::{Database, GemmTable};
pub use crate::engine::{Engine, OpResult, StaticMode};
pub use crate::op::OpSpec;

#[cfg(feature = "python")]
mod py;
