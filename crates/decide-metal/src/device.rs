//! Metal device + command queue + library — singleton, thread-safe.
//!
//! Hazard tracking: every buffer this layer creates ([`DecideDevice::alloc_buffer`],
//! `cached_buffer*`, `register_mmap*`) is `HazardTrackingModeUntracked`,
//! so Metal inserts no dependencies between the dispatches that use them and
//! ordering dependent dispatches is the caller's job. Dispatches on one serial
//! compute encoder run one after another in encode order, and decide encodes
//! each forward pass on one; ordering across encoders needs an `MTLFence`, and
//! across command buffers an `MTLEvent` or a CPU wait.
//!
//! Derived from MicroLLM's `crates/microllm-runtime/src/metal_dev.rs` (see NOTICE).

use anyhow::{ensure, Context, Result};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Function, Library, MTLBuffer, MTLResourceOptions};
use metal::foreign_types::ForeignType;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};
use std::collections::HashMap;

/// A reference to a region within an `MTLBuffer`. Used in place of plain
/// `&Buffer` so that many tensors can share one mmap-spanning `MTLBuffer`
/// (see [`DecideDevice::register_mmap`]) at different byte offsets.
///
/// Clone is cheap: `Buffer` is reference-counted by Metal's ObjC bridge,
/// so `.clone()` just bumps the retain count.
#[derive(Clone)]
pub struct BufferRef {
    pub buf: Buffer,
    /// Byte offset within `buf`. Passed verbatim to
    /// `enc.set_buffer(idx, Some(&self.buf), self.offset)`.
    pub offset: u64,
}

impl BufferRef {
    /// Wrap a whole buffer (offset 0).
    pub fn at_zero(buf: Buffer) -> Self {
        Self { buf, offset: 0 }
    }
}

/// Wrap a `Buffer` reference as a `BufferRef` at offset 0. Equivalent to
/// `BufferRef::at_zero(buf.clone())` but reads cleaner at call sites.
///
/// `Buffer` is reference-counted by Metal's ObjC bridge so `.clone()` just
/// bumps the retain count — no byte copy.
#[inline]
pub fn bref_from(buf: &Buffer) -> BufferRef {
    BufferRef::at_zero(buf.clone())
}

/// Lazily-initialized handle to the system default Metal device.
pub struct DecideDevice {
    device: Device,
    queue: CommandQueue,
    library: Library,
    /// Cache of compiled `ComputePipelineState` objects, keyed by function
    /// name. Pipeline-state creation is *not* free (it links the kernel for
    /// the GPU at first use), and a forward pass dispatches many kernels.
    pipelines: RwLock<HashMap<String, ComputePipelineState>>,
    /// Pool of free `MTLBuffer`s keyed by exact byte length. Recycles
    /// transient kernel outputs across forward passes so steady state
    /// allocates no new buffers. Buffers enter the pool only after the pass's
    /// `wait_until_completed` (see [`Self::flush_step_buffers`]) so the
    /// GPU is guaranteed done with them before reuse.
    free_pool: RwLock<HashMap<u64, Vec<Buffer>>>,
    /// One no-copy `MTLBuffer` per region registered with
    /// [`Self::register_mmap`] or [`Self::register_mmap_lazy`].
    /// [`Self::cached_buffer_with_offset`] resolves a slice inside a region
    /// to `(region buffer, offset)`, so tensors share the region's buffer
    /// instead of getting their own copies. A linear scan over a handful of
    /// regions is trivial overhead.
    mmap_buffers: RwLock<Vec<MmapBuffer>>,
    /// `MTLResidencySet` (macOS 15+) attached to the queue. Pinned resources
    /// won't be evicted from GPU memory under host-side memory pressure.
    /// Created on first use; `None` inside when residency sets are
    /// unavailable. Raw pointer because metal-rs 0.27 doesn't expose
    /// `MTLResidencySet`; lives as long as the process-wide `SHARED` device.
    residency_set: OnceLock<Option<ResidencySet>>,
}

/// Wrapper around a raw `MTLResidencySet*` ObjC pointer. Sendable because
/// it's process-singleton owned and never moved cross-thread once set.
struct ResidencySet(*mut Object);
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

#[derive(Clone)]
struct MmapBuffer {
    buf: Buffer,
    base: usize,
    end: usize,
    /// Added to the residency set (so unregistering must remove it).
    pinned: bool,
}

thread_local! {
    /// Buffers handed out during the current forward step (`alloc_buffer`
    /// transients and `cached_buffer` copies), kept alive by an
    /// extra refcount so they can be returned to the pool
    /// at the post-`wait_until_completed` flush point even if the caller's
    /// locals went out of scope.
    static CURRENT_STEP_BUFFERS: RefCell<Vec<Buffer>> = const { RefCell::new(Vec::new()) };
}

/// The device, or the error that stopped it initialising (kept as a
/// `String` because `anyhow::Error` isn't `Clone`).
static SHARED: OnceLock<Result<DecideDevice, String>> = OnceLock::new();

/// The VM page size: 16 KiB on Apple Silicon.
pub fn page_size() -> usize {
    extern "C" {
        fn getpagesize() -> i32;
    }
    // SAFETY: getpagesize has no preconditions.
    unsafe { getpagesize() as usize }
}

impl DecideDevice {
    /// Get the process-wide singleton. Initializes on first call; concurrent
    /// first calls initialize once, and an init failure is returned to every
    /// caller.
    pub fn shared() -> Result<&'static DecideDevice> {
        SHARED
            .get_or_init(|| Self::new().map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow::anyhow!("Metal device init failed: {e}"))
    }

    fn new() -> Result<Self> {
        let device = Device::system_default().context("no Metal device available")?;
        let queue = device.new_command_queue();
        anyhow::ensure!(
            !crate::METALLIB.is_empty(),
            "embedded metallib is empty — was the Xcode Metal Toolchain installed \
             at build time? Run `xcodebuild -downloadComponent MetalToolchain`."
        );
        let library = device
            .new_library_with_data(crate::METALLIB)
            .map_err(|e| anyhow::anyhow!("loading embedded metallib: {e}"))?;
        Ok(Self {
            device,
            queue,
            library,
            pipelines: RwLock::new(HashMap::new()),
            free_pool: RwLock::new(HashMap::new()),
            mmap_buffers: RwLock::new(Vec::new()),
            residency_set: OnceLock::new(),
        })
    }

    /// Get (lazy-init) the residency set; concurrent first calls create it
    /// once. `None` when residency sets are unavailable.
    fn residency_set_ptr(&self) -> Option<*mut Object> {
        self.residency_set
            .get_or_init(|| self.new_residency_set())
            .as_ref()
            .map(|rs| rs.0)
    }

    /// On macOS 15+ create an `MTLResidencySet` and attach it to the
    /// command queue. On older systems the
    /// `newResidencySetWithDescriptor:error:` selector is missing — we
    /// detect that and return `None`, falling back to default residency.
    fn new_residency_set(&self) -> Option<ResidencySet> {
        // Probe: does MTLDevice respond to newResidencySetWithDescriptor:error:?
        let dev_ptr = self.device.as_ptr() as *mut Object;
        let responds: bool = unsafe {
            let sel_name = sel!(newResidencySetWithDescriptor:error:);
            let r: bool = msg_send![dev_ptr, respondsToSelector: sel_name];
            r
        };
        if !responds {
            return None;
        }
        // Build descriptor: NSClassFromString(@"MTLResidencySetDescriptor")
        // alloc + init, then [device newResidencySetWithDescriptor:err].
        unsafe {
            let cls = metal::objc::runtime::Class::get("MTLResidencySetDescriptor")?;
            let desc: *mut Object = msg_send![cls, alloc];
            let desc: *mut Object = msg_send![desc, init];
            let mut err: *mut Object = std::ptr::null_mut();
            let rs: *mut Object = msg_send![dev_ptr,
                newResidencySetWithDescriptor: desc
                error: &mut err];
            let _: () = msg_send![desc, release];
            if rs.is_null() {
                return None;
            }
            // Attach to queue so all command buffers see it.
            let q_ptr = self.queue.as_ptr() as *mut Object;
            let _: () = msg_send![q_ptr, addResidencySet: rs];
            Some(ResidencySet(rs))
        }
    }

    /// Pin a `Buffer` so the OS won't evict it from GPU memory under
    /// host-side memory pressure. No-op on macOS < 15 or when the
    /// `DECIDE_NO_RESIDENCY` env var is set (debug toggle).
    /// Nothing unpins a buffer pinned here, so pin only process-lifetime
    /// buffers. ([`Self::register_mmap`] pins its regions itself, and
    /// [`Self::unregister_mmap`] unpins them.)
    pub fn pin_buffer(&self, buf: &Buffer) {
        self.pin(buf);
    }

    /// [`Self::pin_buffer`]; returns whether `buf` joined the residency set.
    fn pin(&self, buf: &Buffer) -> bool {
        if std::env::var("DECIDE_NO_RESIDENCY").is_ok() { return false; }
        let Some(rs) = self.residency_set_ptr() else { return false; };
        let alloc_ptr = buf.as_ptr() as *mut Object;
        unsafe {
            let _: () = msg_send![rs, addAllocation: alloc_ptr];
            let _: () = msg_send![rs, commit];
            let _: () = msg_send![rs, requestResidency];
        }
        true
    }

    /// Undo [`Self::pin`]: the residency set releases `buf`.
    fn unpin(&self, buf: &Buffer) {
        let Some(rs) = self.residency_set_ptr() else { return; };
        let alloc_ptr = buf.as_ptr() as *mut Object;
        unsafe {
            let _: () = msg_send![rs, removeAllocation: alloc_ptr];
            let _: () = msg_send![rs, commit];
        }
    }

    /// Allocate (or recycle) a `StorageModeShared` `MTLBuffer` of `size`
    /// bytes for transient kernel outputs. Returned buffers are tracked in
    /// a thread-local "current step" list and returned to the pool by
    /// [`Self::flush_step_buffers`] AFTER the step's
    /// `wait_until_completed` — never sooner, so the GPU is guaranteed
    /// done with them before they're reissued.
    pub fn alloc_buffer(&self, size: u64) -> Buffer {
        let buf = if let Some(b) = self
            .free_pool
            .write()
            .unwrap()
            .get_mut(&size)
            .and_then(|v| v.pop())
        {
            b
        } else {
            self.device.new_buffer(size, MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked)
        };
        // Hold an extra ref in the per-step list so the buffer survives
        // through the post-step flush even if all caller-side references
        // dropped earlier.
        CURRENT_STEP_BUFFERS.with(|cell| cell.borrow_mut().push(buf.clone()));
        buf
    }

    /// Return every buffer handed out during the current step back to the
    /// free pool. Must be called by the forward function AFTER its final
    /// `wait_until_completed` so the GPU is no longer touching the
    /// buffers' backing memory.
    pub fn flush_step_buffers(&self) {
        let bufs = self.take_step_buffers();
        self.recycle_buffers(bufs);
    }

    /// Detach the current thread's step-buffer list WITHOUT recycling —
    /// for pipelined chunks, where a chunk's transients must stay out of
    /// the pool until that chunk's GPU work (and any later chunk reading
    /// its buffers) has completed. Pair with [`Self::recycle_buffers`].
    pub fn take_step_buffers(&self) -> Vec<Buffer> {
        CURRENT_STEP_BUFFERS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }

    /// Return a previously-taken buffer list to the free pool. Caller
    /// guarantees the GPU is done with every buffer in the list.
    pub fn recycle_buffers(&self, bufs: Vec<Buffer>) {
        if bufs.is_empty() {
            return;
        }
        let mut pool = self.free_pool.write().unwrap();
        for buf in bufs {
            pool.entry(buf.length()).or_default().push(buf);
        }
    }

    /// Register a memory-mapped weights region: create one `MTLBuffer` that
    /// aliases the whole region (no copy: the GPU reads the mapping's own
    /// pages), prefault its pages, and pin it resident. Subsequent
    /// [`Self::cached_buffer_with_offset`] calls whose slice falls within
    /// the region return `(region buffer, offset)`, so tensors share this
    /// buffer. Returns the region's buffer.
    ///
    /// `DECIDE_NO_PREFAULT=1` skips the prefault and `DECIDE_NO_RESIDENCY=1`
    /// the pin (debug toggles). Fails if `base` is not page-aligned or the
    /// region is longer than the device's `maxBufferLength`.
    ///
    /// Several regions (e.g. one per weights blob) may be registered;
    /// re-registering the exact same `(base, len)`, with either function,
    /// returns the existing buffer and leaves its mode alone: the first
    /// registration's mode (pinned here, lazy for [`Self::register_mmap_lazy`])
    /// wins.
    ///
    /// # Safety
    ///
    /// `base..base + len` must be a page-aligned, readable mapping whose last
    /// page is readable to its end, as an mmap's tail page is (the buffer's
    /// length is rounded up to whole pages). It must stay mapped and
    /// unchanged until [`Self::unregister_mmap`] has forgotten it and every
    /// GPU use of its buffer has finished.
    pub unsafe fn register_mmap(&self, base: *const u8, len: usize) -> Result<Buffer> {
        // SAFETY: the caller upholds `register_region`'s contract, which is this one.
        unsafe { self.register_region(base, len, true) }
    }

    /// Like [`Self::register_mmap`], but lazy: no prefault and no residency
    /// pin. For the embedding table, which should page in only the rows
    /// that are read.
    ///
    /// # Safety
    ///
    /// As for [`Self::register_mmap`].
    pub unsafe fn register_mmap_lazy(&self, base: *const u8, len: usize) -> Result<Buffer> {
        // SAFETY: as above.
        unsafe { self.register_region(base, len, false) }
    }

    /// Forget a region registered with `register_mmap*` and unpin it, so it
    /// can be unmapped once every GPU use of its buffer has finished. Slices
    /// of it no longer resolve to its buffer. Unknown regions are ignored.
    pub fn unregister_mmap(&self, base: *const u8, len: usize) {
        let (base, end) = (base as usize, base as usize + len);
        let mut regions = self.mmap_buffers.write().unwrap();
        if let Some(i) = regions.iter().position(|mb| mb.base == base && mb.end == end) {
            let region = regions.swap_remove(i);
            if region.pinned {
                self.unpin(&region.buf);
            }
        }
    }

    /// `register_mmap` (`resident`) or `register_mmap_lazy`.
    ///
    /// # Safety
    ///
    /// As for [`Self::register_mmap`].
    unsafe fn register_region(&self, base: *const u8, len: usize, resident: bool) -> Result<Buffer> {
        let base_addr = base as usize;
        let end_addr = base_addr + len;
        let mut regions = self.mmap_buffers.write().unwrap();
        if let Some(region) = regions.iter().find(|mb| mb.base == base_addr && mb.end == end_addr) {
            return Ok(region.buf.clone());
        }
        let page = page_size();
        ensure!(len > 0, "cannot register an empty region");
        ensure!(base_addr % page == 0, "region at {base:p} is not aligned to the {page}-byte page size");
        // A no-copy buffer covers whole pages. Rounding up stays inside the
        // mapping's tail page, which is readable (and zero past the file).
        let buf_len = len.next_multiple_of(page) as u64;
        let max = self.device.max_buffer_length();
        ensure!(
            buf_len <= max,
            "a {buf_len}-byte region exceeds this GPU's maximum buffer length ({max} bytes)"
        );
        if resident {
            // Hint the kernel to bring pages into RAM, then pre-fault every
            // page (unless `DECIDE_NO_PREFAULT` is set). This moves page-fault
            // latency out of the first dispatches that read the weights; the
            // residency-set pin below keeps them resident.
            //
            // posix_madvise FFI: declared inline to avoid pulling in the
            // `libc` crate just for one symbol. POSIX_MADV_WILLNEED = 3 on
            // macOS and Linux. The call is advisory; failure is silent.
            extern "C" {
                fn posix_madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
            }
            const POSIX_MADV_WILLNEED: i32 = 3;
            // SAFETY: the caller guarantees `base..base + len` is mapped.
            unsafe {
                posix_madvise(base as *mut c_void, len, POSIX_MADV_WILLNEED);
            }
            if std::env::var("DECIDE_NO_PREFAULT").is_err() {
                let mut sink: u8 = 0;
                let mut off = 0usize;
                while off < len {
                    // SAFETY: `off < len`, so the read is inside the mapped region.
                    sink ^= unsafe { *base.add(off) };
                    off += page;
                }
                std::hint::black_box(sink);
            }
        }
        // No copy: on a 16 GB machine, duplicating ~8 GB of weights into
        // Metal-owned memory is not an option. The buffer's pages are the
        // mapping's file-backed pages. (MicroLLM saw 100x+ latency outliers
        // on memory-pressured 16 GB Macs when the OS evicted such pages
        // despite a residency-set pin; suspect that first if steady-state
        // latency spikes.)
        //
        // Untracked, like every buffer from this layer (see the module docs).
        // The weights are read-only once registered, so there is no hazard to
        // track, and skipping tracking saves CPU encoding time on every
        // dispatch that reads them.
        let opt = MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        // metal-rs's `new_buffer_with_bytes_no_copy` wraps the returned
        // pointer in a non-null `Buffer` unchecked, so a nil result would be
        // undefined behaviour there. Send the message directly and check it.
        let dev_ptr = self.device.as_ptr() as *mut Object;
        let no_deallocator: *mut Object = std::ptr::null_mut();
        // SAFETY: the caller guarantees the page-aligned region is mapped for
        // `buf_len` bytes (see above); a nil deallocator leaves the memory to
        // its owner.
        let raw: *mut MTLBuffer = unsafe {
            msg_send![dev_ptr, newBufferWithBytesNoCopy: base as *mut c_void
                                                length: buf_len
                                               options: opt
                                           deallocator: no_deallocator]
        };
        ensure!(!raw.is_null(), "Metal refused a no-copy buffer over the {len}-byte region at {base:p}");
        // SAFETY: `raw` is non-null and, from a `new…` method, already
        // retained for us; `Buffer` takes over that reference.
        let buf = unsafe { Buffer::from_ptr(raw) };
        // Pin the whole region in GPU memory. macOS 15+ only (no-op
        // otherwise, or with `DECIDE_NO_RESIDENCY=1`). Prevents page eviction
        // under memory pressure — without this, a busy browser/IDE can page
        // weights out between requests and cause large latency spikes.
        let pinned = resident && self.pin(&buf);
        regions.push(MmapBuffer { buf: buf.clone(), base: base_addr, end: end_addr, pinned });
        Ok(buf)
    }

    /// Like [`Self::cached_buffer`] but returns `(buffer, offset)`. When
    /// the slice lies within a region registered with `register_mmap*`,
    /// returns that region's buffer with the slice's byte offset —
    /// zero-copy. Otherwise falls back to the copying
    /// [`Self::cached_buffer`] path with offset 0.
    ///
    /// The scan over registered regions is linear; with a handful of
    /// regions this is a few pointer compares.
    ///
    /// Callers should pass the returned `offset` to `set_buffer(_, _, offset)`.
    pub fn cached_buffer_with_offset(&self, bytes: &[u8]) -> (metal::Buffer, u64) {
        let ptr = bytes.as_ptr() as usize;
        let end = ptr + bytes.len();
        let regions = self.mmap_buffers.read().unwrap();
        for mb in regions.iter() {
            if ptr >= mb.base && end <= mb.end {
                return (mb.buf.clone(), (ptr - mb.base) as u64);
            }
        }
        drop(regions);
        (self.cached_buffer(bytes), 0)
    }

    /// Copy `bytes` into a new shared `MTLBuffer`, kept alive until the next
    /// [`Self::flush_step_buffers`] (so it outlives the command buffer that
    /// uses it, even if the caller drops it first) and then recycled. For
    /// transient host data; slices of a registered region should go through
    /// [`Self::cached_buffer_with_offset`], which does not copy them.
    pub fn cached_buffer(&self, bytes: &[u8]) -> metal::Buffer {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        );
        CURRENT_STEP_BUFFERS.with(|cell| cell.borrow_mut().push(buf.clone()));
        buf
    }

    pub fn device(&self) -> &Device { &self.device }
    pub fn command_queue(&self) -> &CommandQueue { &self.queue }
    pub fn library(&self) -> &Library { &self.library }

    pub fn function(&self, name: &str) -> Result<Function> {
        self.library
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!("function {name}: {e}"))
    }

    /// Get (or build + cache) the `ComputePipelineState` for a named kernel.
    /// Pipeline creation links the kernel for the GPU and is the slow path —
    /// caching here is the difference between ~10 ms and ~10 µs per dispatch.
    pub fn pipeline(&self, name: &str) -> Result<ComputePipelineState> {
        if let Some(p) = self.pipelines.read().unwrap().get(name) {
            return Ok(p.clone());
        }
        let func = self.function(name)?;
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("pipeline {name}: {e}"))?;
        self.pipelines
            .write()
            .unwrap()
            .insert(name.to_string(), p.clone());
        Ok(p)
    }

    /// Get (or build + cache) the `ComputePipelineState` for a kernel that
    /// declares `[[function_constant(N)]]` u32 constants. Each call with a
    /// different `(name, consts)` pair builds a distinct PSO; PSOs are cached
    /// by a (name, consts) key so repeated calls with the same constants
    /// reuse the same pipeline.
    ///
    /// Constants are specialized at pipeline-build time, so the compiler can
    /// unroll loops bounded by them.
    pub fn pipeline_with_consts_u32(
        &self,
        name: &str,
        consts: &[(u32, u32)], // (index, value)
    ) -> Result<ComputePipelineState> {
        // Cache key encodes the constants alongside the kernel name.
        let mut key = String::with_capacity(name.len() + 8 * consts.len() + 4);
        key.push_str(name);
        for (i, v) in consts {
            key.push('|');
            key.push_str(&i.to_string());
            key.push('=');
            key.push_str(&v.to_string());
        }
        if let Some(p) = self.pipelines.read().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let fcv = metal::FunctionConstantValues::new();
        for (idx, val) in consts {
            let v = *val;
            fcv.set_constant_value_at_index(
                (&v as *const u32) as *const std::ffi::c_void,
                metal::MTLDataType::UInt,
                *idx as u64,
            );
        }
        let func = self
            .library
            .get_function(name, Some(fcv))
            .map_err(|e| anyhow::anyhow!("function {name} with consts: {e}"))?;
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("pipeline {name} with consts: {e}"))?;
        self.pipelines.write().unwrap().insert(key, p.clone());
        Ok(p)
    }
}
