//! Per-kernel GPU timing for perf work: a process-global table of timings
//! keyed by kernel name.
//!
//! [`is_enabled`] reports whether `DECIDE_PROFILE_PERKERNEL` is set. A caller
//! that profiles runs each dispatch in its own command buffer, commits,
//! waits, and passes the wall-clock time to [`record`]; [`dump`] prints the
//! table to stderr sorted by total time, and [`reset`] clears it.
//!
//! Splitting the work this way destroys cross-dispatch GPU/CPU overlap — the
//! *absolute* numbers are upper bounds. The *relative* numbers (which kernel
//! dominates by aggregate time) are exactly what we want for tuning.
//!
//! Derived from MicroLLM's `crates/microllm-runtime/src/perkernel_profiler.rs` (see NOTICE).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Default, Clone, Copy)]
pub struct KernelStat {
    pub count: u64,
    pub total_ns: u128,
    pub min_ns: u64,
    pub max_ns: u64,
}

static PROFILER: Mutex<Option<HashMap<String, KernelStat>>> = Mutex::new(None);
static ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn is_enabled() -> bool {
    // The profiler env var is read once at first call and cached process-wide.
    // Callers query this on every dispatch; going through `std::env::var`
    // each time costs ~1us via the libc env mutex.
    *ENABLED.get_or_init(|| std::env::var("DECIDE_PROFILE_PERKERNEL").is_ok())
}

pub fn record(name: &str, elapsed_ns: u64) {
    let mut g = PROFILER.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let stat = map.entry(name.to_string()).or_insert(KernelStat {
        count: 0,
        total_ns: 0,
        min_ns: u64::MAX,
        max_ns: 0,
    });
    stat.count += 1;
    stat.total_ns += elapsed_ns as u128;
    if elapsed_ns < stat.min_ns {
        stat.min_ns = elapsed_ns;
    }
    if elapsed_ns > stat.max_ns {
        stat.max_ns = elapsed_ns;
    }
}

/// Reset to empty.
pub fn reset() {
    let mut g = PROFILER.lock().unwrap();
    *g = None;
}

/// Print all kernels sorted by total time descending. Writes to stderr.
pub fn dump(banner: &str) {
    let g = PROFILER.lock().unwrap();
    let Some(map) = g.as_ref() else {
        eprintln!("[perkernel] {banner}: (empty)");
        return;
    };
    if map.is_empty() {
        eprintln!("[perkernel] {banner}: (empty)");
        return;
    }
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    let total_ns: u128 = entries.iter().map(|(_, s)| s.total_ns).sum();
    eprintln!(
        "[perkernel] {banner}: total {:.2} ms across {} dispatches in {} unique kernels",
        total_ns as f64 / 1e6,
        entries.iter().map(|(_, s)| s.count).sum::<u64>(),
        entries.len(),
    );
    eprintln!(
        "  {:<48} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6}",
        "kernel", "count", "avg_us", "min_us", "max_us", "total_ms", "pct",
    );
    for (name, stat) in entries.iter() {
        let avg_us = stat.total_ns as f64 / stat.count as f64 / 1e3;
        let total_ms = stat.total_ns as f64 / 1e6;
        let pct = stat.total_ns as f64 * 100.0 / total_ns as f64;
        eprintln!(
            "  {:<48} {:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.3} {:>5.1}%",
            name,
            stat.count,
            avg_us,
            stat.min_ns as f64 / 1e3,
            stat.max_ns as f64 / 1e3,
            total_ms,
            pct,
        );
    }
}
