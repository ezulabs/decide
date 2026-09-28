//! GPU execution time of the command buffers the forward pass commits, from
//! `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` (metal-rs 0.27 has no binding
//! for them, so through `msg_send`). [`Forward`](crate::forward::Forward)
//! records every command buffer it has waited on; the
//! [`Engine`](crate::engine::Engine) takes the total around one forward pass.
//!
//! The total is kept per thread: a forward pass commits and waits on the
//! thread that called it, so a request's total never mixes with another
//! thread's work.
use decide_metal::metal::foreign_types::ForeignTypeRef;
use decide_metal::metal::objc::runtime::Object;
use decide_metal::metal::objc::{msg_send, sel, sel_impl};
use decide_metal::metal::CommandBufferRef;
use std::cell::Cell;

thread_local! {
    /// GPU seconds recorded on this thread since the last [`take`], and
    /// whether any command buffer reported a time.
    static TOTAL: Cell<(f64, bool)> = const { Cell::new((0.0, false)) };
}

/// Add the GPU time of `cmd`, a command buffer that has completed.
pub fn record(cmd: &CommandBufferRef) {
    let ptr = cmd.as_ptr() as *mut Object;
    // SAFETY: every MTLCommandBuffer has these two CFTimeInterval (double)
    // properties; once the buffer has completed they hold its GPU start and end.
    let (start, end): (f64, f64) = unsafe { (msg_send![ptr, GPUStartTime], msg_send![ptr, GPUEndTime]) };
    // Zeros (no GPU timestamps) or a failed buffer: nothing to add.
    if start > 0.0 && end >= start {
        TOTAL.with(|t| t.set((t.get().0 + (end - start), true)));
    }
}

/// The GPU time recorded on this thread since the last call, in ms, or
/// `None` if no command buffer reported one; resets the total.
pub fn take() -> Option<f64> {
    let (seconds, any) = TOTAL.with(|t| t.replace((0.0, false)));
    any.then_some(seconds * 1e3)
}
