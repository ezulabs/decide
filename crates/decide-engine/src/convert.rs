//! Checkpoint (bf16 safetensors) → decide blobs. See the plan's table for names
//! and layouts; this module is its executable form.

use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

use crate::config::{
    check_config, is_full_attention, ATT_IN_N, CONV_DIM, CONV_K, DV, HEAD_DIM, HIDDEN, INTER,
    KV_DIM, LIN_IN_N, MLP_IN_N, NQ, NV, N_LAYERS, Q_DIM, SWIGLU_BLOCK, VALUE_DIM, VOCAB,
};
use crate::weights::DType;
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub const ALIGN: usize = 16 * 1024;

/// The index file in a converted weights directory, and its format tag.
pub const INDEX_FILE: &str = "decide-index.json";
pub const FORMAT: &str = "decide-weights-v1";
/// The blob holding only the bf16 embedding table.
pub const EMBED_BLOB: &str = "decide-embed.bin";
/// Layers per layer blob; blob `decide-lNN.bin` starts at layer NN.
pub const LAYERS_PER_BLOB: usize = 8;

// Blobs hold host-order bytes that the GPU reads in place; Apple Silicon is
// little-endian, and the index promises little-endian.
const _: () = assert!(cfg!(target_endian = "little"));

/// HF q_proj rows (per head: head_dim query rows then head_dim gate rows) →
/// [all query rows | all gate rows]. Rows are `k` elements of bf16 bits.
pub fn deinterleave_q_gate(src: &[u16], heads: usize, head_dim: usize, k: usize) -> Vec<u16> {
    assert_eq!(src.len(), heads * 2 * head_dim * k, "q_proj size");
    let mut out = Vec::with_capacity(src.len());
    for part in 0..2 {
        for h in 0..heads {
            let start = (h * 2 * head_dim + part * head_dim) * k;
            out.extend_from_slice(&src[start..start + head_dim * k]);
        }
    }
    out
}

/// gate_proj and up_proj rows interleaved in blocks of `block` rows.
pub fn interleave_gate_up(gate: &[u16], up: &[u16], block: usize, k: usize) -> Vec<u16> {
    let n = gate.len() / k;
    assert!(
        up.len() == gate.len() && gate.len() == n * k && n % block == 0,
        "gate/up must be equal [n, k] matrices with n a multiple of {block}"
    );
    let mut out = Vec::with_capacity(2 * gate.len());
    for b in 0..n / block {
        out.extend_from_slice(&gate[b * block * k..(b + 1) * block * k]);
        out.extend_from_slice(&up[b * block * k..(b + 1) * block * k]);
    }
    out
}

#[derive(Default, Debug, serde::Serialize)]
pub struct Report {
    pub tensors: usize,
    pub f16_values: u64,
    pub f16_subnormal: u64,
    pub f16_flushed_to_zero: u64,
    pub f16_overflow: u64,
}

/// bf16 bits → f16 bits (round to nearest even), counting precision loss.
pub fn bf16_to_f16(src: &[u16], report: &mut Report) -> Vec<u16> {
    src.iter()
        .map(|&b| {
            let x = f32::from_bits((b as u32) << 16);
            let h = half::f16::from_f32(x);
            report.f16_values += 1;
            if h.is_infinite() && x.is_finite() {
                report.f16_overflow += 1;
            } else if h == half::f16::ZERO && x != 0.0 {
                report.f16_flushed_to_zero += 1;
            } else if h.classify() == std::num::FpCategory::Subnormal {
                report.f16_subnormal += 1;
            }
            h.to_bits()
        })
        .collect()
}

/// Convert `model_dir` into `out_dir`. Fails if any weight overflows fp16.
///
/// Every output is written to a `.tmp` file first and renamed into place once
/// all of them are complete, the index last. A run that fails before that
/// commit leaves any previous conversion intact (and deletes its `.tmp`
/// blobs), and a process that has the old blobs mapped keeps reading the old
/// files rather than having them truncated under it.
pub fn convert(model_dir: &Path, out_dir: &Path) -> Result<Report> {
    // The two files copied verbatim, read now so that a missing one fails the
    // run before any work rather than during the commit.
    let read = |name: &str| {
        let path = model_dir.join(name);
        fs::read(&path).with_context(|| format!("reading {}", path.display()))
    };
    let tokenizer_json = read("tokenizer.json")?;
    let config_json = read("config.json")?;
    let config_path = model_dir.join("config.json");
    let config: Value =
        serde_json::from_slice(&config_json).with_context(|| format!("parsing {}", config_path.display()))?;
    check_config(&config)
        .with_context(|| format!("{} is not Intern-Decision-4B's config", config_path.display()))?;

    // Map the shards that hold language-model tensors (not the vision tower's).
    let index_path = model_dir.join("model.safetensors.index.json");
    let index = read_json(&index_path)?;
    let weight_map = index["weight_map"]
        .as_object()
        .with_context(|| format!("{} has no weight_map", index_path.display()))?;
    let mut files: Vec<String> = Vec::new();
    let mut shard_of: HashMap<String, usize> = HashMap::new();
    for (name, file) in weight_map {
        if !name.starts_with(SRC_PREFIX) {
            continue;
        }
        let file = file.as_str().with_context(|| format!("weight_map[{name}] is not a file name"))?;
        let shard = match files.iter().position(|f| f == file) {
            Some(i) => i,
            None => {
                files.push(file.to_string());
                files.len() - 1
            }
        };
        shard_of.insert(name.clone(), shard);
    }
    let maps: Vec<Mmap> = files.iter().map(|f| map_file(&model_dir.join(f))).collect::<Result<_>>()?;
    let shards: Vec<SafeTensors> = maps
        .iter()
        .zip(&files)
        .map(|(map, file)| SafeTensors::deserialize(map).with_context(|| format!("parsing {file}")))
        .collect::<Result<_>>()?;
    let ck = Checkpoint { files: &files, shards, shard_of };

    fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    ensure!(
        fs::canonicalize(out_dir)? != fs::canonicalize(model_dir)?,
        "--out must differ from the model directory"
    );
    let mut blob_names = vec![EMBED_BLOB.to_string()];
    blob_names.extend((0..N_LAYERS).step_by(LAYERS_PER_BLOB).map(|l| format!("decide-l{l:02}.bin")));
    let mut out = Writer {
        blobs: blob_names.iter().map(|name| Blob::create(out_dir, name)).collect::<Result<_>>()?,
        tensors: Map::new(),
        report: Report::default(),
        overflowed: Vec::new(),
    };

    let embed = ck.get("embed_tokens.weight")?;
    check_view(&embed, "embed_tokens.weight", &[VOCAB, HIDDEN])?;
    ensure!(embed.dtype() == Dtype::BF16, "embed_tokens.weight is {:?}, expected BF16", embed.dtype());
    out.put(0, "embed".into(), DType::BF16, &[VOCAB, HIDDEN], &[embed.data()])?;
    for layer in 0..N_LAYERS {
        convert_layer(&ck, &mut out, layer, 1 + layer / LAYERS_PER_BLOB)?;
    }
    let last = out.blobs.len() - 1;
    out.put_f32(last, "final_norm", &[HIDDEN], &ck.small_f32("norm.weight", &[HIDDEN])?)?;

    ensure!(
        out.report.f16_overflow == 0,
        "{} weights overflow fp16 (in {}); no index written",
        out.report.f16_overflow,
        out.overflowed.join(", ")
    );

    // Commit. Drop the old index first: from here until the new index lands,
    // the directory must not look loadable.
    let index_file = out_dir.join(INDEX_FILE);
    match fs::remove_file(&index_file) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(e).with_context(|| format!("removing {}", index_file.display()))
        }
        _ => {}
    }
    for blob in &mut out.blobs {
        blob.commit()?;
    }
    for (name, bytes) in [("tokenizer.json", &tokenizer_json), ("config.json", &config_json)] {
        write_then_rename(&out_dir.join(name), bytes)?;
    }
    let index = json!({
        "format": FORMAT,
        "blobs": blob_names,
        "tensors": out.tensors,
        "report": out.report,
    });
    write_then_rename(&index_file, serde_json::to_string_pretty(&index)?.as_bytes())?;
    Ok(out.report)
}

/// Every source name below is relative to this prefix.
const SRC_PREFIX: &str = "model.language_model.";

/// One layer's tensors, in forward-pass order, into blob `blob`.
fn convert_layer(ck: &Checkpoint, out: &mut Writer, i: usize, blob: usize) -> Result<()> {
    let src = |s: &str| format!("layers.{i}.{s}");
    let dst = |s: &str| format!("L{i}.{s}");
    let hidden = [HIDDEN];
    out.put_f32(blob, &dst("in_norm"), &hidden, &ck.small_f32(&src("input_layernorm.weight"), &hidden)?)?;
    if is_full_attention(i) {
        let q = ck.bf16(&src("self_attn.q_proj.weight"), &[2 * Q_DIM, HIDDEN])?;
        let k = ck.bf16(&src("self_attn.k_proj.weight"), &[KV_DIM, HIDDEN])?;
        let v = ck.bf16(&src("self_attn.v_proj.weight"), &[KV_DIM, HIDDEN])?;
        let q_gate = deinterleave_q_gate(&q, NQ, HEAD_DIM, HIDDEN);
        out.put_f16(blob, &dst("w_in"), &[ATT_IN_N, HIDDEN], &[&q_gate, &k, &v])?;
        for norm in ["q_norm", "k_norm"] {
            let w = ck.small_f32(&src(&format!("self_attn.{norm}.weight")), &[HEAD_DIM])?;
            out.put_f32(blob, &dst(norm), &[HEAD_DIM], &w)?;
        }
        let o = ck.bf16(&src("self_attn.o_proj.weight"), &[HIDDEN, Q_DIM])?;
        out.put_f16(blob, &dst("w_out"), &[HIDDEN, Q_DIM], &[&o])?;
    } else {
        let qkv = ck.bf16(&src("linear_attn.in_proj_qkv.weight"), &[CONV_DIM, HIDDEN])?;
        let z = ck.bf16(&src("linear_attn.in_proj_z.weight"), &[VALUE_DIM, HIDDEN])?;
        let b = ck.bf16(&src("linear_attn.in_proj_b.weight"), &[NV, HIDDEN])?;
        let a = ck.bf16(&src("linear_attn.in_proj_a.weight"), &[NV, HIDDEN])?;
        out.put_f16(blob, &dst("w_in"), &[LIN_IN_N, HIDDEN], &[&qkv, &z, &b, &a])?;
        // [CONV_DIM, 1, CONV_K] squeezed: the bytes are already [CONV_DIM, CONV_K].
        let conv = ck.small_f32(&src("linear_attn.conv1d.weight"), &[CONV_DIM, 1, CONV_K])?;
        out.put_f32(blob, &dst("conv_w"), &[CONV_DIM, CONV_K], &conv)?;
        out.put_f32(blob, &dst("a_log"), &[NV], &ck.small_f32(&src("linear_attn.A_log"), &[NV])?)?;
        out.put_f32(blob, &dst("dt_bias"), &[NV], &ck.small_f32(&src("linear_attn.dt_bias"), &[NV])?)?;
        out.put_f32(blob, &dst("gnorm"), &[DV], &ck.small_f32(&src("linear_attn.norm.weight"), &[DV])?)?;
        let o = ck.bf16(&src("linear_attn.out_proj.weight"), &[HIDDEN, VALUE_DIM])?;
        out.put_f16(blob, &dst("w_out"), &[HIDDEN, VALUE_DIM], &[&o])?;
    }
    let post = ck.small_f32(&src("post_attention_layernorm.weight"), &hidden)?;
    out.put_f32(blob, &dst("post_norm"), &hidden, &post)?;
    let gate = ck.bf16(&src("mlp.gate_proj.weight"), &[INTER, HIDDEN])?;
    let up = ck.bf16(&src("mlp.up_proj.weight"), &[INTER, HIDDEN])?;
    let gate_up = interleave_gate_up(&gate, &up, SWIGLU_BLOCK, HIDDEN);
    out.put_f16(blob, &dst("w_gu"), &[MLP_IN_N, HIDDEN], &[&gate_up])?;
    let down = ck.bf16(&src("mlp.down_proj.weight"), &[HIDDEN, INTER])?;
    out.put_f16(blob, &dst("w_down"), &[HIDDEN, INTER], &[&down])?;
    Ok(())
}

/// The checkpoint's language-model shards, with tensors looked up by name.
struct Checkpoint<'a> {
    files: &'a [String],
    shards: Vec<SafeTensors<'a>>,
    /// Full tensor name → index into `files`/`shards`.
    shard_of: HashMap<String, usize>,
}

impl<'a> Checkpoint<'a> {
    /// The tensor `model.language_model.{name}`.
    fn get(&self, name: &str) -> Result<TensorView<'a>> {
        let full = format!("{SRC_PREFIX}{name}");
        let &shard = self
            .shard_of
            .get(&full)
            .with_context(|| format!("the checkpoint index has no tensor {full}"))?;
        self.shards[shard]
            .tensor(&full)
            .with_context(|| format!("tensor {full} is missing from {}", self.files[shard]))
    }

    /// A projection weight as bf16 bits; it must be bf16 with shape `shape`.
    fn bf16(&self, name: &str, shape: &[usize]) -> Result<Cow<'a, [u16]>> {
        let t = self.get(name)?;
        check_view(&t, name, shape)?;
        if t.dtype() != Dtype::BF16 {
            bail!("projection weight {name} is {:?}; decide converts only bf16 projections", t.dtype());
        }
        Ok(u16s(t.data()))
    }

    /// A small tensor (norm weights, conv, A_log, dt_bias), bf16 or f32, as f32.
    fn small_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        check_view(&t, name, shape)?;
        let data = t.data();
        Ok(match t.dtype() {
            Dtype::BF16 => u16s(data).iter().map(|&b| f32::from_bits((b as u32) << 16)).collect(),
            Dtype::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => bail!("{name} is {other:?}; expected bf16 or f32"),
        })
    }
}

fn check_view(t: &TensorView, name: &str, shape: &[usize]) -> Result<()> {
    ensure!(t.shape() == shape, "{name} has shape {:?}; decide expects {shape:?}", t.shape());
    Ok(())
}

/// Little-endian u16s from bytes: borrowed when aligned, copied otherwise.
fn u16s(bytes: &[u8]) -> Cow<'_, [u16]> {
    // SAFETY: every bit pattern is a valid u16.
    match unsafe { bytes.align_to::<u16>() } {
        ([], words, []) => Cow::Borrowed(words),
        _ => Cow::Owned(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
    }
}

/// The output blobs, the index entries written so far, and the report.
struct Writer {
    blobs: Vec<Blob>,
    tensors: Map<String, Value>,
    report: Report,
    /// Tensors with at least one value that overflowed fp16.
    overflowed: Vec<String>,
}

impl Writer {
    /// Append a tensor made of `parts` (concatenated) to blob `blob`.
    fn put(&mut self, blob: usize, name: String, dtype: DType, shape: &[usize], parts: &[&[u8]]) -> Result<()> {
        let bytes: usize = parts.iter().map(|p| p.len()).sum();
        ensure!(
            bytes == dtype.size() * shape.iter().product::<usize>(),
            "{name}: {bytes} bytes do not make a {} {shape:?} tensor",
            dtype.name()
        );
        let b = &mut self.blobs[blob];
        let offset = b.begin_tensor()?;
        for part in parts {
            b.write(part).with_context(|| format!("writing {name} to {}", b.name))?;
        }
        let entry = json!({"blob": b.name, "offset": offset, "dtype": dtype.name(), "shape": shape});
        ensure!(self.tensors.insert(name.clone(), entry).is_none(), "tensor {name} written twice");
        self.report.tensors += 1;
        Ok(())
    }

    /// bf16 rows (concatenated `parts`) → an f16 tensor.
    fn put_f16(&mut self, blob: usize, name: &str, shape: &[usize], parts: &[&[u16]]) -> Result<()> {
        let before = self.report.f16_overflow;
        let converted: Vec<Vec<u16>> = parts.iter().map(|p| bf16_to_f16(p, &mut self.report)).collect();
        if self.report.f16_overflow > before {
            self.overflowed.push(name.to_string());
        }
        let bytes: Vec<&[u8]> = converted.iter().map(|c| as_bytes(c)).collect();
        self.put(blob, name.to_string(), DType::F16, shape, &bytes)
    }

    fn put_f32(&mut self, blob: usize, name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
        self.put(blob, name.to_string(), DType::F32, shape, &[as_bytes(values)])
    }
}

/// Plain-old-data element types whose in-memory bytes are the blob bytes.
trait Pod: Copy {}
impl Pod for u16 {}
impl Pod for f32 {}

fn as_bytes<T: Pod>(values: &[T]) -> &[u8] {
    // SAFETY: u16 and f32 have no padding, so every byte is initialised.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// One output blob, written to `<name>.tmp` until [`Blob::commit`]; an
/// uncommitted blob deletes its `.tmp` when dropped.
struct Blob {
    name: String,
    tmp: PathBuf,
    path: PathBuf,
    file: BufWriter<File>,
    len: usize,
    committed: bool,
}

impl Blob {
    fn create(dir: &Path, name: &str) -> Result<Self> {
        let path = dir.join(name);
        let tmp = dir.join(format!("{name}.tmp"));
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let file = BufWriter::with_capacity(8 << 20, file);
        Ok(Self { name: name.to_string(), tmp, path, file, len: 0, committed: false })
    }

    /// Zero-pad to the next `ALIGN` boundary.
    fn pad(&mut self) -> Result<()> {
        static ZEROS: [u8; ALIGN] = [0; ALIGN];
        let pad = self.len.next_multiple_of(ALIGN) - self.len;
        self.write(&ZEROS[..pad])
    }

    /// Pad, and return the new tensor's offset.
    fn begin_tensor(&mut self) -> Result<usize> {
        self.pad()?;
        Ok(self.len)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes)?;
        self.len += bytes.len();
        Ok(())
    }

    /// Pad the tail (so the file is whole pages), flush, and rename into place.
    fn commit(&mut self) -> Result<()> {
        self.pad()?;
        self.file.flush().with_context(|| format!("writing {}", self.tmp.display()))?;
        fs::rename(&self.tmp, &self.path).with_context(|| format!("renaming {}", self.tmp.display()))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        if !self.committed {
            // A failed conversion's partial output; up to ~2 GB each.
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// The ANE weight set's parameters, the `"ane"` object of its index (see
/// [`convert_ane`]); the Python side (`reference/ane_convert.py`) builds the
/// models with the same values and records them in `ane-models.json`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AneConfig {
    /// SwiGLU intermediate dims `[0, n_inter)` are on the ANE; the GPU keeps
    /// the rest (`w_gu` rows `2·n_inter..`, `w_down` columns `n_inter..`).
    pub n_inter: usize,
    /// Rows per ANE predict (the models' fixed shape).
    pub chunk_rows: usize,
    /// The models' `w_up` is pre-scaled by this; the engine adds `partial / out_scale`.
    pub out_scale: f32,
    /// Layers `0..layers` are split; the rest keep their full MLP on the GPU.
    pub layers: usize,
}

/// The ANE weight set: `weights_dir`'s tensors with layers `0..layers`' MLP
/// weights cut down to the GPU's slice (so the GPU never maps the ANE's part),
/// the embedding blob hard-linked, `tokenizer.json`/`config.json` copied, and
/// an `"ane"` object in the index. The per-layer models (`L{l}.mlmodelc`)
/// come from `reference/ane_convert.py` into the same directory.
pub fn convert_ane(weights_dir: &Path, out_dir: &Path, cfg: &AneConfig) -> Result<Report> {
    ensure!(cfg.n_inter % 128 == 0 && cfg.n_inter > 0 && cfg.n_inter < INTER, "n_inter {} must be a multiple of 128 below {INTER}", cfg.n_inter);
    // The last layer's MLP runs on gathered answer rows (the pruned path), not
    // on the whole chunk the ANE session is bound to: it must stay whole.
    ensure!(cfg.layers < N_LAYERS, "{} split layers; at most {} (the last layer stays on the GPU)", cfg.layers, N_LAYERS - 1);
    ensure!(cfg.chunk_rows > 0 && cfg.chunk_rows % 8 == 0, "chunk_rows {} must be a positive multiple of 8", cfg.chunk_rows);
    let index_path = weights_dir.join(INDEX_FILE);
    let index = read_json(&index_path)?;
    ensure!(index["format"].as_str() == Some(FORMAT), "{} has format {}", index_path.display(), index["format"]);
    ensure!(index.get("ane").is_none(), "{} is already an ANE weight set", weights_dir.display());
    let src_blobs: Vec<String> = index["blobs"]
        .as_array()
        .context("the index has no blob list")?
        .iter()
        .map(|b| b.as_str().map(str::to_string).context("a blob name is not a string"))
        .collect::<Result<_>>()?;
    let maps: HashMap<&str, Mmap> =
        src_blobs.iter().filter(|b| *b != EMBED_BLOB).map(|b| Ok((b.as_str(), map_file(&weights_dir.join(b))?))).collect::<Result<_>>()?;
    let tensors = index["tensors"].as_object().context("the index has no tensor table")?;

    fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    ensure!(fs::canonicalize(out_dir)? != fs::canonicalize(weights_dir)?, "--out must differ from the weights directory");
    let blob_names: Vec<String> = src_blobs.iter().filter(|b| *b != EMBED_BLOB).cloned().collect();
    let mut out = Writer {
        blobs: blob_names.iter().map(|name| Blob::create(out_dir, name)).collect::<Result<_>>()?,
        tensors: Map::new(),
        report: Report::default(),
        overflowed: Vec::new(),
    };
    // The index is ordered as `convert` wrote it (tensor by tensor per blob), so
    // copying in that order keeps each blob's layout.
    for (name, entry) in tensors {
        let blob_name = entry["blob"].as_str().with_context(|| format!("tensor {name}: bad blob"))?;
        let offset = entry["offset"].as_u64().with_context(|| format!("tensor {name}: bad offset"))? as usize;
        let dtype = match entry["dtype"].as_str() {
            Some("f16") => DType::F16,
            Some("f32") => DType::F32,
            Some("bf16") => DType::BF16,
            other => bail!("tensor {name}: bad dtype {other:?}"),
        };
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .and_then(|d| d.iter().map(|v| v.as_u64().map(|v| v as usize)).collect())
            .with_context(|| format!("tensor {name}: bad shape"))?;
        if blob_name == EMBED_BLOB {
            ensure!(name == "embed", "unexpected tensor {name} in {EMBED_BLOB}");
            out.tensors.insert(name.clone(), entry.clone());
            out.report.tensors += 1;
            continue;
        }
        let blob = blob_names.iter().position(|b| b == blob_name).with_context(|| format!("tensor {name}: blob {blob_name}"))?;
        let nbytes = dtype.size() * shape.iter().product::<usize>();
        let src = &maps[blob_name][offset..offset + nbytes];
        let layer = name.strip_prefix('L').and_then(|s| s.split('.').next()).and_then(|l| l.parse::<usize>().ok());
        let split = layer.is_some_and(|l| l < cfg.layers);
        if split && name.ends_with(".w_gu") {
            ensure!(shape == [MLP_IN_N, HIDDEN] && dtype == DType::F16, "{name} is {dtype:?} {shape:?}");
            let rows = MLP_IN_N - 2 * cfg.n_inter;
            out.put(blob, name.clone(), DType::F16, &[rows, HIDDEN], &[&src[2 * cfg.n_inter * HIDDEN * 2..]])?;
        } else if split && name.ends_with(".w_down") {
            ensure!(shape == [HIDDEN, INTER] && dtype == DType::F16, "{name} is {dtype:?} {shape:?}");
            let cols = INTER - cfg.n_inter;
            let parts: Vec<&[u8]> = (0..HIDDEN).map(|r| &src[(r * INTER + cfg.n_inter) * 2..(r + 1) * INTER * 2]).collect();
            out.put(blob, name.clone(), DType::F16, &[HIDDEN, cols], &parts)?;
        } else {
            out.put(blob, name.clone(), dtype, &shape, &[src])?;
        }
    }
    let embed_out = out_dir.join(EMBED_BLOB);
    let _ = fs::remove_file(&embed_out);
    if fs::hard_link(weights_dir.join(EMBED_BLOB), &embed_out).is_err() {
        fs::copy(weights_dir.join(EMBED_BLOB), &embed_out).with_context(|| format!("copying {EMBED_BLOB}"))?;
    }
    let index_file = out_dir.join(INDEX_FILE);
    match fs::remove_file(&index_file) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(e).with_context(|| format!("removing {}", index_file.display()))
        }
        _ => {}
    }
    for blob in &mut out.blobs {
        blob.commit()?;
    }
    for name in ["tokenizer.json", "config.json"] {
        let bytes = fs::read(weights_dir.join(name)).with_context(|| format!("reading {name}"))?;
        write_then_rename(&out_dir.join(name), &bytes)?;
    }
    let mut all_blobs = vec![EMBED_BLOB.to_string()];
    all_blobs.extend(blob_names);
    let new_index = json!({
        "format": FORMAT,
        "blobs": all_blobs,
        "tensors": out.tensors,
        "report": out.report,
        "ane": cfg,
        "source": { "index": index_path.display().to_string(), "report": index["report"] },
    });
    write_then_rename(&index_file, serde_json::to_string_pretty(&new_index)?.as_bytes())?;
    Ok(out.report)
}

fn write_then_rename(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", path.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: the checkpoint is input only; nothing writes or truncates it
    // while the conversion runs.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))
}
