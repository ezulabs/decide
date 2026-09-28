//! Plain f32/f64 reference implementations of everything the engine runs on
//! the GPU. Slow on purpose; kernel tests compare against these.

use crate::config::*;

pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
pub fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }

/// Row-wise RMSNorm over `h` columns; `add_one` applies the zero-centred (1 + w).
pub fn rmsnorm(x: &[f32], w: &[f32], h: usize, add_one: bool) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (row, o) in x.chunks(h).zip(out.chunks_mut(h)) {
        let ms = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / h as f64;
        let inv = 1.0 / ((ms as f32) + EPS).sqrt();
        for i in 0..h {
            let wv = if add_one { 1.0 + w[i] } else { w[i] };
            o[i] = row[i] * inv * wv;
        }
    }
    out
}

/// y[m][n] = Σ_k x[m][k] · w[n][k]
pub fn gemm_nt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            y[i * n + j] = xr.iter().zip(wr).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32;
        }
    }
    y
}

/// SwiGLU over an interleaved gate|up row (blocks of SWIGLU_BLOCK): [m, n2] → [m, n2/2].
pub fn swiglu_interleaved(y: &[f32], m: usize, n2: usize) -> Vec<f32> {
    let (n, blk) = (n2 / 2, SWIGLU_BLOCK);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let (b, w) = (j / blk, j % blk);
            let g = y[i * n2 + b * 2 * blk + w];
            let u = y[i * n2 + b * 2 * blk + blk + w];
            out[i * n + j] = silu(g) * u;
        }
    }
    out
}

/// Causal depthwise conv (kernel 4) + SiLU over the qkv columns of `p` (row
/// stride `ldp`), with `prev` = the 3 raw rows before this chunk, oldest first.
/// Returns (q, k, v): q = l2norm(q)·DK^-½, k = l2norm(k), per head.
pub fn conv_silu_l2(p: &[f32], ldp: usize, t: usize, prev: &[f32], w: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let raw = |s: isize, c: usize| if s >= 0 { p[s as usize * ldp + c] } else { prev[(3 + s) as usize * CONV_DIM + c] };
    let mut y = vec![0.0f32; t * CONV_DIM];
    for s in 0..t {
        for c in 0..CONV_DIM {
            let acc: f64 = (0..CONV_K).map(|j| w[c * CONV_K + j] as f64 * raw(s as isize - 3 + j as isize, c) as f64).sum();
            y[s * CONV_DIM + c] = silu(acc as f32);
        }
    }
    let l2 = |seg: &[f32], scale: f32| -> Vec<f32> {
        let ss = seg.iter().map(|&x| x as f64 * x as f64).sum::<f64>();
        let inv = 1.0 / ((ss as f32) + 1e-6).sqrt();
        seg.iter().map(|&x| x * inv * scale).collect()
    };
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..t {
        let row = &y[s * CONV_DIM..(s + 1) * CONV_DIM];
        for h in 0..NK {
            q.extend(l2(&row[h * DK..(h + 1) * DK], (DK as f32).powf(-0.5)));
        }
        for h in 0..NK {
            k.extend(l2(&row[KEY_DIM + h * DK..KEY_DIM + (h + 1) * DK], 1.0));
        }
        v.extend_from_slice(&row[2 * KEY_DIM..]);
    }
    (q, k, v)
}

/// The conv state after a chunk of `t` rows: the 3 raw qkv rows that end it,
/// oldest first. Row i is chunk position s = t − 3 + i, taken from `p[s]` (row
/// stride `ldp`) if s ≥ 0, else from `prev[3 + s]` (the state before the chunk).
pub fn conv_state_update(p: &[f32], ldp: usize, t: usize, prev: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; 3 * CONV_DIM];
    for i in 0..3 {
        let s = t as isize - 3 + i as isize;
        for c in 0..CONV_DIM {
            out[i * CONV_DIM + c] = if s >= 0 { p[s as usize * ldp + c] } else { prev[(3 + s) as usize * CONV_DIM + c] };
        }
    }
    out
}

/// Sequential gated delta rule. q,k [t,16,128]; v [t,32,128]; a,b read from
/// rows of stride `ldab` (value head h at column h); state [32][128 dv][128 dk]
/// updated in place. Returns o [t,32,128].
#[allow(clippy::too_many_arguments)]
pub fn gated_delta(q: &[f32], k: &[f32], v: &[f32], a: &[f32], b: &[f32], ldab: usize,
                   a_log: &[f32], dt_bias: &[f32], state: &mut [f32], t: usize) -> Vec<f32> {
    let mut o = vec![0.0; t * NV * DV];
    for s in 0..t {
        for h in 0..NV {
            let hk = h / (NV / NK);
            let g = (-(a_log[h].exp()) * softplus(a[s * ldab + h] + dt_bias[h])).exp();
            let beta = sigmoid(b[s * ldab + h]);
            let kt = &k[(s * NK + hk) * DK..(s * NK + hk + 1) * DK];
            let qt = &q[(s * NK + hk) * DK..(s * NK + hk + 1) * DK];
            for dv in 0..DV {
                let row = &mut state[(h * DV + dv) * DK..(h * DV + dv + 1) * DK];
                let mut kv = 0.0f64;
                for dk in 0..DK {
                    row[dk] *= g;
                    kv += row[dk] as f64 * kt[dk] as f64;
                }
                let delta = (v[(s * NV + h) * DV + dv] - kv as f32) * beta;
                let mut out = 0.0f64;
                for dk in 0..DK {
                    row[dk] += kt[dk] * delta;
                    out += row[dk] as f64 * qt[dk] as f64;
                }
                o[(s * NV + h) * DV + dv] = out as f32;
            }
        }
    }
    o
}

/// Per (row, value head): RMSNorm over 128 with weight w (no +1), times silu(z).
pub fn gated_rmsnorm_silu(o: &[f32], z: &[f32], ldz: usize, w: &[f32], t: usize) -> Vec<f32> {
    let mut y = vec![0.0; t * VALUE_DIM];
    for s in 0..t {
        for h in 0..NV {
            let seg = &o[(s * NV + h) * DV..(s * NV + h + 1) * DV];
            let ms = seg.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / DV as f64;
            let inv = 1.0 / ((ms as f32) + EPS).sqrt();
            for d in 0..DV {
                y[s * VALUE_DIM + h * DV + d] = seg[d] * inv * w[d] * silu(z[s * ldz + h * DV + d]);
            }
        }
    }
    y
}

/// Zero-centred per-head RMSNorm over 256 dims, then RoPE on dims [0,64) at
/// absolute position `pos` (pairs i, i+32; fp32 angle = pos·inv_freq[i]).
pub fn head_norm_rope(seg: &[f32], w: &[f32], pos: usize) -> Vec<f32> {
    let ms = seg.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / HEAD_DIM as f64;
    let inv = 1.0 / ((ms as f32) + EPS).sqrt();
    let mut x: Vec<f32> = (0..HEAD_DIM).map(|d| seg[d] * inv * (1.0 + w[d])).collect();
    let f = inv_freq();
    for i in 0..ROT_DIM / 2 {
        let ang = pos as f32 * f[i];
        let (sn, cs) = ((ang as f64).sin() as f32, (ang as f64).cos() as f32);
        let (a, b) = (x[i], x[i + ROT_DIM / 2]);
        x[i] = a * cs - b * sn;
        x[i + ROT_DIM / 2] = b * cs + a * sn;
    }
    x
}

/// Causal GQA attention (16 q heads over 4 kv heads, head_dim 256, scale 1/16)
/// for query rows at absolute positions `qpos`, keys/values [npos, 4, 256];
/// output [rows, 4096] multiplied by sigmoid(gate) ([rows, 4096]).
pub fn attention(q: &[f32], qpos: &[usize], kc: &[f32], vc: &[f32], gate: &[f32]) -> Vec<f32> {
    let rows = qpos.len();
    let mut out = vec![0.0; rows * Q_DIM];
    for (s, &pos) in qpos.iter().enumerate() {
        for h in 0..NQ {
            let kvh = h / (NQ / NKV);
            let qv = &q[(s * NQ + h) * HEAD_DIM..(s * NQ + h + 1) * HEAD_DIM];
            let scores: Vec<f64> = (0..=pos).map(|j| {
                let kr = &kc[(j * NKV + kvh) * HEAD_DIM..(j * NKV + kvh + 1) * HEAD_DIM];
                qv.iter().zip(kr).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() / 16.0
            }).collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..HEAD_DIM {
                let acc: f64 = (0..=pos).map(|j| e[j] * vc[(j * NKV + kvh) * HEAD_DIM + d] as f64).sum();
                out[s * Q_DIM + h * HEAD_DIM + d] = (acc / z) as f32 * sigmoid(gate[s * Q_DIM + h * HEAD_DIM + d]);
            }
        }
    }
    out
}

/// Final zero-centred RMSNorm of each row, then logits against its candidate
/// embedding rows (bf16 table rows already widened to f32).
pub fn final_head(x: &[f32], w: &[f32], candidates: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    let normed = rmsnorm(x, w, HIDDEN, true);
    candidates.iter().enumerate().map(|(r, cands)| {
        let h = &normed[r * HIDDEN..(r + 1) * HIDDEN];
        cands.iter().map(|e| h.iter().zip(e).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_by_hand() {
        // mean(x²) = (9 + 16)/2 = 12.5 → inv ≈ 0.2828427; (1 + w) = 2 for both.
        let y = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 2, true);
        assert!((y[0] - 3.0 * 2.0 / 12.5f32.sqrt()).abs() < 1e-5);
        assert!((y[1] - 4.0 * 2.0 / 12.5f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn swiglu_pairs_blocks() {
        let n2 = 2 * SWIGLU_BLOCK;
        let mut y = vec![0.0; n2];
        y[3] = 1.0; // gate col 3
        y[SWIGLU_BLOCK + 3] = 2.0; // up col 3
        let out = swiglu_interleaved(&y, 1, n2);
        assert!((out[3] - silu(1.0) * 2.0).abs() < 1e-6);
        assert!(out.iter().enumerate().all(|(i, &v)| i == 3 || v == 0.0));
    }

    #[test]
    fn delta_rule_with_unit_gate_is_a_delta_update() {
        // One step, beta = 1 (b huge), g = 1 (A_log = -inf → exp(A_log) = 0).
        let (mut state, t) = (vec![0.0; NV * DV * DK], 1);
        let mut k = vec![0.0; NK * DK];
        k[0] = 1.0;
        let q = k.clone();
        let mut v = vec![0.0; NV * DV];
        v[5] = 3.0; // head 0, column 5
        let (a, b) = (vec![0.0; NV], vec![50.0; NV]);
        let o = gated_delta(&q, &k, &v, &a, &b, NV, &[f32::NEG_INFINITY; NV], &[0.0; NV], &mut state, t);
        assert!((o[5] - 3.0).abs() < 1e-5);      // S = k δᵀ → o = S q = δ = v
        assert!((state[5 * DK] - 3.0).abs() < 1e-5);
    }
}
