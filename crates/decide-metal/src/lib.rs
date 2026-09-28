//! Metal device layer for decide: one device and queue, the embedded metallib,
//! cached pipelines, and zero-copy buffers over mmapped weights.
//! Derived from MicroLLM's runtime (see NOTICE).
// objc 0.2.7's `sel!`/`msg_send!` expand a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

pub mod device;
pub mod gemm_configs;
pub mod profile;
pub mod source_hash;

pub use device::{bref_from, BufferRef, DecideDevice};
pub use metal;

/// Every `kernels/*.metal`, compiled at build time.
pub const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/decide.metallib"));

/// Content hash of the sources [`METALLIB`] was compiled from
/// (`source_hash::hash_sources` over `kernels/`). Compare it with a fresh
/// hash of the checkout to catch a stale metallib; the engine reports it.
pub const METALLIB_SOURCES_HASH: &str = include_str!(concat!(env!("OUT_DIR"), "/metallib_sources_hash.txt"));
