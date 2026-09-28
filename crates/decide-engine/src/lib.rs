//! Intern-Decision-4B on Metal: weights, kernels, forward pass, prefix snapshot,
//! decision head, and the public [`engine::Engine`] over them.

pub mod config;
pub mod convert;
pub mod cpu_ref;
pub mod engine;
pub mod forward;
pub mod gpu;
pub mod gpu_time;
pub mod head;
pub mod kernels;
pub mod testutil;
pub mod weights;
