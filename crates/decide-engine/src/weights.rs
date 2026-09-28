//! Converted weights (see [`crate::convert`]), mapped and handed to Metal
//! without copying. Layer blobs are prefaulted and pinned resident; the
//! embedding blob is lazy, so only the rows that are read page in.

use crate::config::check_config;
use crate::convert::{AneConfig, ALIGN, EMBED_BLOB, FORMAT, INDEX_FILE};
use anyhow::{bail, ensure, Context, Result};
use decide_metal::DecideDevice;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    BF16,
}

impl DType {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    /// The name `decide-index.json` uses.
    pub fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::BF16 => "bf16",
        }
    }

    fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "f16" => DType::F16,
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            _ => bail!("unknown dtype {name:?}"),
        })
    }
}

/// A tensor inside a registered blob: bind it with
/// `set_buffer(i, Some(&t.buf), t.offset)`.
///
/// `buf` aliases the blob's mapping, which [`Weights`] unmaps when it drops
/// (after draining the device's command queue). A `Buffer` cloned out of a
/// `Tensor` does not keep the mapping alive: using such a clone after the
/// `Weights` drops, or on another command queue, is the caller's
/// responsibility.
pub struct Tensor {
    pub buf: decide_metal::metal::Buffer,
    pub offset: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

pub struct Weights {
    // Fields drop in declaration order: the tensors' buffers, which alias the
    // mappings, go first.
    tensors: HashMap<String, Entry>,
    blobs: Vec<Blob>,
    /// Set for an ANE weight set (`decide ane-convert`): layers `0..layers`
    /// hold only the GPU's MLP slice and need the ANE models next to them.
    ane: Option<AneConfig>,
}

struct Entry {
    tensor: Tensor,
    /// Index into `Weights::blobs`.
    blob: usize,
}

struct Blob {
    name: String,
    map: Mmap,
}

impl Weights {
    /// mmap every blob and register it with Metal. Layer blobs are prefaulted
    /// and pinned resident; the embedding blob is lazy.
    ///
    /// Nothing may modify or truncate the blobs while they are open
    /// (`decide convert` renames new files into place, which is safe).
    /// Dropping the `Weights` waits for the device's command queue to drain
    /// before it unmaps them; GPU work that uses the tensors through any
    /// other queue must finish first.
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join(INDEX_FILE);
        ensure!(index_path.exists(), "{} does not exist; run `decide convert`", index_path.display());
        let index = read_json(&index_path)?;
        ensure!(
            index["format"].as_str() == Some(FORMAT),
            "{} has format {}; expected {FORMAT}",
            index_path.display(),
            index["format"]
        );
        let config_path = dir.join("config.json");
        check_config(&read_json(&config_path)?)
            .with_context(|| format!("{} is not Intern-Decision-4B's config", config_path.display()))?;
        let dev = DecideDevice::shared()?;

        // Built up in place so that, if a step fails, dropping it
        // unregisters the blobs registered so far.
        let ane = match index.get("ane") {
            Some(v) => Some(
                serde_json::from_value::<AneConfig>(v.clone())
                    .with_context(|| format!("{}: bad \"ane\" section", index_path.display()))?,
            ),
            None => None,
        };
        let mut weights = Weights { tensors: HashMap::new(), blobs: Vec::new(), ane };
        let mut bufs = Vec::new();
        for name in index["blobs"].as_array().context("the index has no blob list")? {
            let name = name.as_str().context("a blob name is not a string")?;
            let path = dir.join(name);
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: nothing modifies or truncates the blob while it is
            // mapped (see above).
            let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
            // SAFETY: `map` is a page-aligned, read-only mapping of
            // `map.len()` bytes whose tail page is readable to its end. It
            // stays mapped and unchanged until `Weights` drops, and `drop`
            // drains the device's queue and unregisters it before unmapping
            // (work on other queues is the caller's, see above).
            let buf = unsafe {
                if name == EMBED_BLOB {
                    dev.register_mmap_lazy(map.as_ptr(), map.len())
                } else {
                    dev.register_mmap(map.as_ptr(), map.len())
                }
            }
            .with_context(|| format!("registering {name} with Metal"))?;
            weights.blobs.push(Blob { name: name.to_string(), map });
            bufs.push(buf);
        }

        for (name, entry) in index["tensors"].as_object().context("the index has no tensor table")? {
            let bad = |field: &str| format!("tensor {name}: missing or bad {field}");
            let blob_name = entry["blob"].as_str().with_context(|| bad("blob"))?;
            let blob = weights
                .blobs
                .iter()
                .position(|b| b.name == blob_name)
                .with_context(|| format!("tensor {name} is in {blob_name}, which is not in the blob list"))?;
            let offset = entry["offset"].as_u64().with_context(|| bad("offset"))?;
            let dtype = DType::from_name(entry["dtype"].as_str().with_context(|| bad("dtype"))?)
                .with_context(|| bad("dtype"))?;
            let shape = entry["shape"]
                .as_array()
                .and_then(|dims| dims.iter().map(|d| d.as_u64().map(|d| d as usize)).collect::<Option<Vec<_>>>())
                .with_context(|| bad("shape"))?;
            ensure!(offset % ALIGN as u64 == 0, "tensor {name}: offset {offset} is not {ALIGN}-byte aligned");
            let end = shape
                .iter()
                .try_fold(dtype.size(), |n, &d| n.checked_mul(d))
                .and_then(|n| n.checked_add(offset as usize));
            ensure!(
                end.is_some_and(|end| end <= weights.blobs[blob].map.len()),
                "tensor {name} runs past the end of {blob_name}"
            );
            let tensor = Tensor { buf: bufs[blob].clone(), offset, dtype, shape };
            weights.tensors.insert(name.clone(), Entry { tensor, blob });
        }
        Ok(weights)
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        Ok(&self.entry(name)?.tensor)
    }

    /// The ANE split this weight set was converted for, if any.
    pub fn ane(&self) -> Option<&AneConfig> {
        self.ane.as_ref()
    }

    /// CPU view of a tensor's bytes (tests, the head's candidate rows).
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let Entry { tensor, blob } = self.entry(name)?;
        let start = tensor.offset as usize;
        Ok(&self.blobs[*blob].map[start..start + tensor.nbytes()])
    }

    /// Every tensor, in no particular order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors.iter().map(|(name, entry)| (name.as_str(), &entry.tensor))
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).with_context(|| format!("the weights have no tensor {name}"))
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        let Ok(dev) = DecideDevice::shared() else { return };
        // Drain the device's queue first: an empty command buffer completes
        // only after those committed before it, so no command buffer still
        // in flight can read the mappings once they are unmapped.
        let cb = dev.command_queue().new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
        // Then, before the fields drop, the device forgets (and unpins) each
        // region, so no later slice can resolve to a buffer over memory that
        // is about to be unmapped.
        for blob in &self.blobs {
            dev.unregister_mmap(blob.map.as_ptr(), blob.map.len());
        }
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}
