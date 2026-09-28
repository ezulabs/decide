//! Dispatch wrappers: one struct per kernel family (one `kernels/*.metal` file
//! each) holding its pipelines, and one method per Metal kernel. Methods only
//! encode; the caller commits. Buffers are passed as [`Arg`]s, `(buffer, byte
//! offset)` pairs, and each family's `#[repr(C)]` parameter structs match its
//! Metal structs field for field. Reach a kernel as `k.<family>.<method>(…)`.
use decide_metal::metal::{Buffer, ComputeCommandEncoderRef};
use decide_metal::DecideDevice;

mod ane;
mod attn;
mod conv;
mod delta;
mod embed;
mod gemm;
mod gnorm;
mod head;
mod norm;
mod rope;

pub use ane::Ane;
pub use attn::{Attn, AttnImpl};
pub use conv::Conv;
pub use delta::{Delta, DeltaImpl};
pub use embed::Embed;
pub use gemm::{Epilogue, Gemm, GemmImpl};
pub use gnorm::GNorm;
pub use head::Head;
pub use norm::Norm;
pub use rope::Rope;

/// A buffer and a byte offset into it.
pub type Arg<'a> = (&'a Buffer, u64);

fn buf(enc: &ComputeCommandEncoderRef, i: u64, a: Arg) { enc.set_buffer(i, Some(a.0), a.1); }
fn bytes<T>(enc: &ComputeCommandEncoderRef, i: u64, v: &T) {
    enc.set_bytes(i, std::mem::size_of::<T>() as u64, v as *const T as *const std::ffi::c_void);
}

/// Every pipeline the engine dispatches, by kernel family.
pub struct Kernels {
    pub ane: Ane,
    pub embed: Embed,
    pub norm: Norm,
    pub gemm: Gemm,
    pub conv: Conv,
    pub delta: Delta,
    pub gnorm: GNorm,
    pub rope: Rope,
    pub attn: Attn,
    pub head: Head,
}

impl Kernels {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self {
            ane: Ane::new(dev)?,
            embed: Embed::new(dev)?,
            norm: Norm::new(dev)?,
            gemm: Gemm::new(dev)?,
            conv: Conv::new(dev)?,
            delta: Delta::new(dev)?,
            gnorm: GNorm::new(dev)?,
            rope: Rope::new(dev)?,
            attn: Attn::new(dev)?,
            head: Head::new(dev)?,
        })
    }
}
