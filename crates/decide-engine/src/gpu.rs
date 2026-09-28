//! Buffers and one-shot command submission, for tests and engine setup.
use decide_metal::metal::{Buffer, ComputeCommandEncoderRef, MTLCommandBufferStatus, MTLResourceOptions};
use decide_metal::DecideDevice;

pub struct Gpu { pub dev: &'static DecideDevice }

impl Gpu {
    pub fn new() -> anyhow::Result<Self> { Ok(Self { dev: DecideDevice::shared()? }) }

    pub fn buffer(&self, bytes: usize) -> Buffer {
        self.dev.device().new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let buf = self.buffer(std::mem::size_of_val(data));
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut T, data.len()) };
        buf
    }
    pub fn f32(&self, v: &[f32]) -> Buffer { self.upload(v) }
    pub fn u32(&self, v: &[u32]) -> Buffer { self.upload(v) }
    pub fn f16(&self, v: &[f32]) -> Buffer {
        self.upload(&v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect::<Vec<u16>>())
    }
    pub fn bf16(&self, v: &[f32]) -> Buffer {
        self.upload(&v.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect::<Vec<u16>>())
    }
    pub fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        assert!(n * 4 <= buf.length() as usize, "read_f32: {n} floats past the buffer end");
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
    }
    pub fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
        assert!(n * 2 <= buf.length() as usize, "read_f16: {n} halves past the buffer end");
        unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, n) }
            .iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
    }
    /// Encode with `f` into one serial compute encoder, commit, wait.
    /// Panics if the command buffer did not complete (a GPU fault).
    pub fn run(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) {
        let cmd = self.dev.command_queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "command buffer failed");
    }
}
