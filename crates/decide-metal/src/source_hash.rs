//! A content hash of the Metal sources the metallib was built from.
//!
//! Shared by `build.rs` (which records it) and the crate (which exposes it
//! as [`crate::METALLIB_SOURCES_HASH`]) via `#[path]`, so both sides hash
//! exactly the same way. Cargo's build-script cache is not keyed by the
//! source *path*: a checkout building into another checkout's `target/` leaves
//! it running a metallib compiled from foreign sources, silently.
//!
//! The set is every `kernels/*.metal` the build compiles (top level only)
//! and every `kernels/**/*.h` header any
//! of them may include. FNV-1a 64 over `relative path NUL contents NUL`, in
//! sorted path order.

use std::fs;
use std::path::{Path, PathBuf};

/// The files that make up the metallib: sorted `(relative path, absolute path)`.
pub fn metallib_sources(kernels_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    collect(kernels_dir, kernels_dir, true, &mut out);
    out.sort();
    out
}

fn collect(root: &Path, dir: &Path, top: bool, out: &mut Vec<(String, PathBuf)>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("kernels dir {} is unreadable: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(root, &path, false, out);
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str());
        let wanted = match ext {
            Some("metal") => top,
            Some("h") => true,
            _ => false,
        };
        if wanted {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            out.push((rel, path));
        }
    }
}

/// FNV-1a 64 of the sources, as 16 lowercase hex digits.
pub fn hash_sources(kernels_dir: &Path) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (rel, path) in metallib_sources(kernels_dir) {
        let contents = fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        feed(rel.as_bytes());
        feed(&[0]);
        feed(&contents);
        feed(&[0]);
    }
    format!("{h:016x}")
}
