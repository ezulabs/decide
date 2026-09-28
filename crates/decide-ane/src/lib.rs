//! The Neural Engine side of decide's GPU+ANE MLP split: a safe wrapper over the
//! Swift shim in `swift/ane.swift`.
//!
//! Per layer, one compiled Core ML model computes the ANE's slice of the MLP
//! for a fixed chunk of `chunk_rows` rows: `partial = swiglu(h·Wg^T, h·Wu^T)·Wd^T`
//! over the first `n_inter` intermediate dims, with `w_up` pre-scaled so the
//! partial is `out_scale`× larger than the true value (the ANE's absolute
//! error floor); the GPU adds `partial / out_scale`. Rows are handed over in
//! chunk slots: slot `s` is rows `[s·chunk_rows, (s+1)·chunk_rows)` of the
//! bound `h` (input, fp16) and `partial` (output, fp16) buffers.
//!
//! Synchronization is an `MTLSharedEvent` the caller owns: a job waits for
//! `wait_value` (the GPU signals it once the normed rows are in `h`) and
//! signals `signal_value` when the partial is written (the GPU's next command
//! buffer waits for it at its head). Jobs fire in order of their wait values.
use anyhow::{anyhow, bail, ensure, Result};
use metal::foreign_types::ForeignTypeRef;
use metal::SharedEventRef;
use std::ffi::{c_char, c_void, CString};
use std::path::Path;

mod ffi {
    use std::ffi::{c_char, c_void};
    extern "C" {
        pub fn ane_open(dir: *const c_char, layers: u32, chunk_rows: u32, in_dim: u32, out_dim: u32, err: *mut c_char, err_len: usize) -> *mut c_void;
        pub fn ane_close(handle: *mut c_void);
        pub fn ane_load_ms(handle: *mut c_void, layer: u32) -> f64;
        pub fn ane_bind(handle: *mut c_void, event: *mut c_void, h: *mut c_void, h_stride: usize, out: *mut c_void, out_stride: usize, slots: u32, err: *mut c_char, err_len: usize) -> i32;
        pub fn ane_warm_up(handle: *mut c_void, err: *mut c_char, err_len: usize) -> i32;
        pub fn ane_enqueue(handle: *mut c_void, layer: u32, chunks: u32, wait_value: u64, signal_value: u64) -> i32;
        pub fn ane_wait_idle(handle: *mut c_void, timeout_s: f64) -> i32;
        pub fn ane_error(handle: *mut c_void, err: *mut c_char, err_len: usize) -> i32;
        pub fn ane_stats(handle: *mut c_void, predicts: *mut u64, seconds: *mut f64);
    }
}

const ERR_LEN: usize = 1024;

fn err_buf() -> Vec<c_char> {
    vec![0; ERR_LEN]
}

fn err_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The per-layer ANE models of one weight set, loaded on `.cpuAndNeuralEngine`.
pub struct AneSession {
    handle: *mut c_void,
    layers: usize,
    chunk_rows: usize,
    slots: usize,
}

// The Swift session is used from the engine's thread and its own serial queue only.
unsafe impl Send for AneSession {}

impl AneSession {
    /// Load `dir/L0.mlmodelc … L{layers-1}.mlmodelc`, each taking fp16
    /// `[chunk_rows, in_dim]` and returning fp16 `[chunk_rows, out_dim]`.
    /// Loads are sequential (the ANE compiler service serializes them anyway).
    pub fn open(dir: &Path, layers: usize, chunk_rows: usize, in_dim: usize, out_dim: usize) -> Result<Self> {
        ensure!(layers > 0 && chunk_rows > 0, "an ANE session needs at least one layer and one row");
        let dir_c = CString::new(dir.to_string_lossy().as_bytes())?;
        let mut err = err_buf();
        // SAFETY: the C strings and the error buffer outlive the call.
        let handle = unsafe {
            ffi::ane_open(dir_c.as_ptr(), layers as u32, chunk_rows as u32, in_dim as u32, out_dim as u32, err.as_mut_ptr(), ERR_LEN)
        };
        if handle.is_null() {
            bail!("loading the ANE models in {}: {}", dir.display(), err_string(&err));
        }
        Ok(Self { handle, layers, chunk_rows, slots: 0 })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    /// Milliseconds loading layer `l`'s model took.
    pub fn load_ms(&self, l: usize) -> f64 {
        // SAFETY: a live handle.
        unsafe { ffi::ane_load_ms(self.handle, l as u32) }
    }

    /// Bind the event and the buffers: `h` and `partial` are the CPU addresses
    /// of shared Metal buffers holding at least `slots · chunk_rows` rows of
    /// fp16 with the given row strides (in elements). The caller keeps the
    /// buffers and the event alive while the session is bound.
    ///
    /// # Safety
    /// `h` and `partial` must stay valid (and not be freed) until the session
    /// is dropped or rebound, and the GPU/ANE ordering described in the module
    /// docs must hold (the caller encodes the signals and waits).
    pub unsafe fn bind(&mut self, event: &SharedEventRef, h: *mut c_void, h_stride: usize, partial: *mut c_void,
                       partial_stride: usize, slots: usize) -> Result<()> {
        ensure!(slots > 0, "an ANE session needs at least one chunk slot");
        let mut err = err_buf();
        let rc = ffi::ane_bind(self.handle, event.as_ptr() as *mut c_void, h, h_stride, partial, partial_stride, slots as u32,
                               err.as_mut_ptr(), ERR_LEN);
        ensure!(rc == 0, "binding the ANE session: {}", err_string(&err));
        self.slots = slots;
        Ok(())
    }

    /// Run every model once on slot 0 (whatever `h` holds), so the ANE
    /// programs are resident before the first request.
    pub fn warm_up(&self) -> Result<()> {
        ensure!(self.slots > 0, "warm_up before bind");
        let mut err = err_buf();
        // SAFETY: bound session; the error buffer outlives the call.
        let rc = unsafe { ffi::ane_warm_up(self.handle, err.as_mut_ptr(), ERR_LEN) };
        ensure!(rc == 0, "ANE warm-up: {}", err_string(&err));
        self.take_error()
    }

    /// Register one job: when the event reaches `wait_value`, run layer `l`'s
    /// model on chunk slots `0..chunks` and set the event to `signal_value`.
    pub fn enqueue(&self, l: usize, chunks: usize, wait_value: u64, signal_value: u64) -> Result<()> {
        ensure!(l < self.layers, "ANE job for layer {l} of {}", self.layers);
        ensure!((1..=self.slots).contains(&chunks), "ANE job with {chunks} chunks; {} slots are bound", self.slots);
        ensure!(signal_value > wait_value, "an ANE job must signal a value above the one it waits for");
        // SAFETY: bound session.
        let rc = unsafe { ffi::ane_enqueue(self.handle, l as u32, chunks as u32, wait_value, signal_value) };
        ensure!(rc == 0, "enqueueing an ANE job (rc {rc})");
        Ok(())
    }

    /// Wait until every job has run (bounded by `timeout_s`) and report the
    /// first error any of them hit.
    pub fn finish(&self, timeout_s: f64) -> Result<()> {
        // SAFETY: a live handle.
        let rc = unsafe { ffi::ane_wait_idle(self.handle, timeout_s) };
        ensure!(rc == 0, "ANE jobs still pending after {timeout_s} s");
        self.take_error()
    }

    /// The first error since the last call, if any.
    pub fn take_error(&self) -> Result<()> {
        let mut err = err_buf();
        // SAFETY: the error buffer outlives the call.
        let rc = unsafe { ffi::ane_error(self.handle, err.as_mut_ptr(), ERR_LEN) };
        if rc != 0 {
            return Err(anyhow!("ANE: {}", err_string(&err)));
        }
        Ok(())
    }

    /// Predicts run so far and the seconds they took (cumulative).
    pub fn stats(&self) -> (u64, f64) {
        let (mut n, mut s) = (0u64, 0f64);
        // SAFETY: a live handle; out-pointers are valid.
        unsafe { ffi::ane_stats(self.handle, &mut n, &mut s) };
        (n, s)
    }
}

impl Drop for AneSession {
    fn drop(&mut self) {
        // Let in-flight jobs finish before their arrays' memory goes away.
        let _ = self.finish(30.0);
        // SAFETY: the handle came from ane_open and is released once.
        unsafe { ffi::ane_close(self.handle) };
    }
}
