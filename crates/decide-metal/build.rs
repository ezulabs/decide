//! Build-time Metal kernel compilation.
//!
//! This script compiles every `kernels/*.metal` source under the workspace
//! root into a single `decide.metallib` blob in `OUT_DIR`, with `kernels/`
//! on the include path so kernels can share headers such as `common.h`.
//! The crate then `include_bytes!`s the blob so the final binary is
//! self-contained.
//!
//! decide is GPU-only: there is no CPU path to fall back to, so a missing
//! Metal Toolchain fails the build just like a broken kernel does.
//!
//! Derived from MicroLLM's `crates/microllm-runtime/build.rs` (see NOTICE).

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[path = "src/source_hash.rs"]
mod source_hash;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent().unwrap().parent().unwrap()
        .join("kernels");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/source_hash.rs");
    // The directory entry catches added and removed files; the per-file
    // entries emitted below name every kernel source and header explicitly.
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    // Headers in subdirectories (kernels/steel/*.h) are included by the
    // top-level sources; name them too, and record what the metallib was
    // built from (see src/source_hash.rs).
    for (_, path) in source_hash::metallib_sources(&kernels_dir) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    fs::write(out_dir.join("metallib_sources_hash.txt"), source_hash::hash_sources(&kernels_dir))
        .expect("cannot write metallib_sources_hash.txt");

    let metallib_path = out_dir.join("decide.metallib");

    // A broken kernel must FAIL THE BUILD. Warning + writing an EMPTY
    // metallib + exit 0 would ship a binary whose every pipeline lookup
    // fails at runtime — silent corruption — and could cache a transient
    // bad metallib under parallel test compilation.
    let mut air_files: Vec<PathBuf> = Vec::new();
    let mut paths: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("kernels dir {} is unreadable: {e}", kernels_dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    // read_dir order is unspecified; sorting links the .air files in the
    // same order on every build, so the metallib is reproducible.
    paths.sort();

    for path in paths {
        let ext = path.extension().and_then(|e| e.to_str());
        if matches!(ext, Some("metal") | Some("h")) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        if ext == Some("metal") {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let air = out_dir.join(format!("{stem}.air"));
            let output = Command::new("xcrun")
                .args(["-sdk", "macosx", "metal", "-c"])
                .arg("-I")
                .arg(&kernels_dir)
                .arg(&path)
                .arg("-o")
                .arg(&air)
                .output();
            match output {
                Ok(o) if o.status.success() => {
                    // A successful compile's stderr holds its warnings;
                    // forward them, or cargo swallows them.
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    for line in stderr.lines().filter(|l| !l.trim().is_empty()) {
                        println!("cargo:warning={line}");
                    }
                    air_files.push(air);
                }
                Ok(o) => {
                    panic!(
                        "Metal kernel compile FAILED for {} ({}):\n{}",
                        path.display(),
                        o.status,
                        String::from_utf8_lossy(&o.stderr),
                    );
                }
                Err(e) => {
                    panic!(
                        "cannot run `xcrun metal`: {e} — install the Xcode Metal \
                         Toolchain via `xcodebuild -downloadComponent MetalToolchain`."
                    );
                }
            }
        }
    }

    assert!(
        !air_files.is_empty(),
        "no .metal sources found in {} — broken checkout?",
        kernels_dir.display()
    );

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&metallib_path);
    let output = cmd.output().expect("cannot run `xcrun metallib`");
    assert!(
        output.status.success(),
        "metallib link FAILED ({}):\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}
