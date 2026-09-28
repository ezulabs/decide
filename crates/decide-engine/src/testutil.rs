//! Helpers shared by the kernel tests.
use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn rng(seed: u64) -> StdRng { StdRng::seed_from_u64(seed) }

pub fn uniform(r: &mut StdRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| r.gen_range(-scale..scale)).collect()
}

/// Round through fp16, so the CPU reference sees exactly what the GPU sees.
pub fn f16_round(v: &[f32]) -> Vec<f32> { v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() }
pub fn bf16_round(v: &[f32]) -> Vec<f32> { v.iter().map(|&x| half::bf16::from_f32(x).to_f32()).collect() }

/// Passes when `|g − w| ≤ atol + rtol·|w|` holds element-wise; a NaN fails.
/// On failure it panics naming the worst element. On success it prints, to
/// the test's captured stderr (`--nocapture` shows it), the largest error and
/// the share of its tolerance that the worst element used.
// `!(a <= b)` rather than `a > b`, so that a NaN counts as a failure.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst: Option<(usize, f32)> = None;
    let (mut max_err, mut max_used) = (0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let (err, tol) = ((g - w).abs(), atol + rtol * w.abs());
        let over = err - tol;
        if !(over <= 0.0) && worst.is_none_or(|(_, o)| !(over <= o)) {
            worst = Some((i, over));
        }
        if err > 0.0 {
            max_err = max_err.max(err);
            max_used = max_used.max(err / tol);
        }
    }
    if let Some((i, _)) = worst {
        panic!("{name}: element {i}: got {} want {} (atol {atol}, rtol {rtol})", got[i], want[i]);
    }
    eprintln!("{name}: max |err| {max_err:.2e}, worst element at {:.1}% of tolerance", 100.0 * max_used);
}
