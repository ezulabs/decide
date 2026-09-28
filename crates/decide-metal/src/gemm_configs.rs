//! Tile-config menus and dispatch geometry for `kernels/gemm_fast.metal`.
//!
//! The two menus mirror `GEMM_TILE_MENU` and `GEMM_SMALL_MENU` in the Metal
//! source: the same ids in the same order. The kernel reads the id from
//! function constant 1 and specialises the tile at pipeline-build time.
//! [`recommend`] holds the per-shape choice measured on an M1 Pro; [`plan`] turns it into the
//! kernel name, function constants, grid, threadgroup size and threadgroup
//! memory the engine has to bind.

/// A threadgroup-tiled config for the `gemm_nt_fast*` entry points.
///
/// `bm × bn` output tile, `bk`-wide k-chunks, `wm × wn` simdgroups (each
/// computing `bm/wm × bn/wn`), `pf` = 0 plain (two barriers, MLX style),
/// 1 register prefetch of the next chunk, 2 double-buffered threadgroup memory;
/// `g` fragments per fragment group; `lm` load mode (0 packed vector operands,
/// 1 pair-adjacent A words + element-wise B). See the Metal source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCfg {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub wm: u32,
    pub wn: u32,
    pub pf: u32,
    pub g: u32,
    pub lm: u32,
}

#[allow(clippy::too_many_arguments)]
const fn t(bm: u32, bn: u32, bk: u32, wm: u32, wn: u32, pf: u32, g: u32, lm: u32) -> TileCfg {
    TileCfg { bm, bn, bk, wm, wn, pf, g, lm }
}

/// Mirrors `GEMM_TILE_MENU` (index = config id; `menus_match_the_metal_source`
/// checks it). `engine` entries are used by [`RECOMMENDED`]; `lab` entries
/// exist for `gemm_lab` sweeps.
pub const TILE_CFGS: [TileCfg; 28] = [
    t(64, 32, 32, 2, 1, 0, 4, 0), //  0 lab
    t(64, 32, 32, 2, 1, 0, 2, 0), //  1 lab
    t(64, 32, 32, 2, 1, 0, 4, 1), //  2 engine
    t(64, 32, 32, 2, 1, 0, 2, 1), //  3 lab
    t(64, 32, 32, 2, 1, 1, 4, 0), //  4 lab
    t(64, 32, 32, 2, 1, 1, 2, 1), //  5 lab
    t(64, 32, 32, 2, 1, 2, 2, 1), //  6 lab
    t(64, 32, 64, 2, 1, 0, 4, 1), //  7 lab
    t(64, 32, 64, 2, 1, 0, 2, 1), //  8 lab
    t(64, 64, 32, 2, 2, 0, 4, 0), //  9 lab
    t(64, 64, 32, 2, 2, 0, 2, 1), // 10 engine
    t(64, 64, 32, 2, 2, 1, 2, 1), // 11 lab
    t(32, 32, 32, 2, 2, 0, 2, 1), // 12 engine
    t(64, 32, 32, 2, 2, 0, 2, 1), // 13 engine
    t(64, 32, 32, 4, 1, 0, 2, 1), // 14 lab
    t(32, 64, 32, 2, 2, 0, 2, 1), // 15 lab
    t(64, 32, 16, 2, 1, 0, 2, 1), // 16 lab
    t(64, 32, 16, 2, 1, 0, 2, 0), // 17 lab
    t(128, 32, 32, 4, 1, 0, 2, 1), // 18 lab
    t(128, 64, 32, 4, 2, 0, 2, 1), // 19 lab
    t(64, 64, 32, 1, 2, 0, 2, 1), // 20 lab
    t(32, 32, 32, 1, 1, 0, 2, 1), // 21 lab
    t(64, 32, 32, 2, 1, 2, 4, 0), // 22 lab
    t(32, 64, 32, 1, 2, 0, 2, 1), // 23 lab
    t(64, 64, 32, 2, 1, 0, 2, 1), // 24 lab
    t(64, 64, 32, 2, 1, 0, 4, 0), // 25 lab
    t(64, 32, 64, 2, 1, 0, 4, 0), // 26 lab
    t(32, 32, 32, 2, 2, 0, 4, 0), // 27 lab
];

/// A config for `gemm_nt_small` (M ≤ 8·tm): each simdgroup owns `tn·8`
/// columns and all `tm·8` rows; `nsg` simdgroups per threadgroup; `ks`
/// simdgroups of a column group split K and reduce in threadgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallCfg {
    pub tm: u32,
    pub tn: u32,
    pub nsg: u32,
    pub ks: u32,
}

const fn s(tm: u32, tn: u32, nsg: u32, ks: u32) -> SmallCfg {
    SmallCfg { tm, tn, nsg, ks }
}

/// Mirrors `GEMM_SMALL_MENU` (index = config id; `engine` / `lab` as above).
pub const SMALL_CFGS: [SmallCfg; 28] = [
    s(2, 4, 1, 1),              //  0 lab
    s(2, 4, 2, 1),              //  1 lab
    s(2, 4, 4, 1),              //  2 lab
    s(2, 4, 4, 4),              //  3 engine
    s(2, 4, 2, 2),              //  4 lab
    s(2, 4, 8, 4),              //  5 lab
    s(2, 2, 2, 1),              //  6 lab
    s(2, 2, 4, 1),              //  7 lab
    s(1, 4, 1, 1),              //  8 lab
    s(1, 4, 2, 1),              //  9 lab
    s(1, 4, 4, 1),              // 10 lab
    s(1, 4, 4, 4),              // 11 engine
    s(1, 4, 8, 4),              // 12 lab
    s(1, 2, 4, 1),              // 13 lab
    s(4, 4, 2, 1),              // 14 lab
    s(4, 4, 4, 4),              // 15 lab
    s(2, 8, 2, 1),              // 16 lab
    s(2, 8, 4, 4),              // 17 lab
    s(1, 8, 2, 1),              // 18 lab
    s(1, 8, 4, 4),              // 19 lab
    s(1, 4, 8, 8),              // 20 lab
    s(2, 4, 8, 8),              // 21 engine
    s(1, 2, 4, 4),              // 22 lab
    s(1, 2, 8, 8),              // 23 engine
    s(2, 2, 4, 4),              // 24 engine
    s(2, 2, 8, 8),              // 25 lab
    s(1, 2, 8, 4),              // 26 lab
    s(2, 2, 8, 4),              // 27 lab
];

/// Which kernel (and config id) to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKernel {
    Tile(u32),
    Small(u32),
}

/// Everything the engine binds for one GEMM dispatch, besides the buffers and
/// `GemmParams`. Function constants: `(0, epilogue)`, `(1, cfg)`, `(2, order)`.
#[derive(Clone, Copy, Debug)]
pub struct GemmDispatch {
    pub kernel: &'static str,
    pub cfg: u32,
    /// Grid order for `gemm_nt_fast`: 1 = threadgroup x walks M tiles (the
    /// M tiles of one W stripe run back to back, so W streams from DRAM once).
    pub order: u32,
    pub grid: (u64, u64),
    pub threads: u64,
    pub smem_bytes: u64,
}

impl GemmDispatch {
    pub fn function_constants(&self, epilogue: u32) -> [(u32, u32); 3] {
        [(0, epilogue), (1, self.cfg), (2, self.order)]
    }
}

fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

impl TileCfg {
    pub fn threads(&self) -> u64 {
        (self.wm * self.wn * 32) as u64
    }
    /// One entry point per threadgroup size: `max_total_threads_per_threadgroup`
    /// sets the register budget, so each size gets the exact value.
    pub fn kernel_name(&self) -> &'static str {
        match self.threads() {
            32 => "gemm_nt_fast_tg32",
            64 => "gemm_nt_fast",
            128 => "gemm_nt_fast_tg128",
            256 => "gemm_nt_fast_tg256",
            _ => panic!("no gemm_nt_fast entry point for {} threads", self.threads()),
        }
    }
    pub fn smem_bytes(&self) -> u64 {
        let ld = self.bk + 8;
        let buf = (self.bm + self.bn) * ld;
        let nbuf = if self.pf == 2 { 2 } else { 1 };
        (buf * nbuf * 2) as u64
    }
    /// SwiGLU pairs fragments 16 columns apart inside one simdgroup: WN ≤ 2.
    pub fn supports_swiglu(&self) -> bool {
        self.wn <= 2
    }
    pub fn supports(&self, k: usize, epilogue: u32) -> bool {
        k as u32 % self.bk == 0 && (epilogue != 3 || self.supports_swiglu())
    }
    pub fn dispatch(&self, id: u32, m: usize, n: usize, order: u32) -> GemmDispatch {
        let tm = ceil_div(m as u64, self.bm as u64);
        let tn = ceil_div(n as u64, self.bn as u64);
        GemmDispatch {
            kernel: self.kernel_name(),
            cfg: id,
            order,
            grid: if order != 0 { (tm, tn) } else { (tn, tm) },
            threads: self.threads(),
            smem_bytes: self.smem_bytes(),
        }
    }
}

impl SmallCfg {
    pub fn threads(&self) -> u64 {
        (self.nsg * 32) as u64
    }
    pub fn max_m(&self) -> usize {
        (self.tm * 8) as usize
    }
    pub fn smem_bytes(&self) -> u64 {
        ((self.ks - 1) * (self.nsg / self.ks) * 32 * self.tm * self.tn * 8) as u64
    }
    pub fn cols_per_threadgroup(&self) -> u64 {
        ((self.nsg / self.ks) * self.tn * 8) as u64
    }
    /// SwiGLU needs whole 32-column gate|up blocks per simdgroup.
    pub fn supports_swiglu(&self) -> bool {
        self.tn % 4 == 0
    }
    pub fn supports(&self, m: usize, k: usize, epilogue: u32) -> bool {
        let k = k as u32;
        m <= self.max_m()
            && k % self.ks == 0
            && (k / self.ks) % 32 == 0
            && (epilogue != 3 || self.supports_swiglu())
    }
    pub fn dispatch(&self, id: u32, n: usize) -> GemmDispatch {
        GemmDispatch {
            kernel: "gemm_nt_small",
            cfg: id,
            order: 1,
            grid: (ceil_div(n as u64, self.cols_per_threadgroup()), 1),
            threads: self.threads(),
            smem_bytes: self.smem_bytes(),
        }
    }
}

/// Dispatch geometry for an explicit kernel choice.
pub fn dispatch_for(kernel: GemmKernel, m: usize, n: usize, order: u32) -> GemmDispatch {
    match kernel {
        GemmKernel::Tile(id) => TILE_CFGS[id as usize].dispatch(id, m, n, order),
        GemmKernel::Small(id) => SMALL_CFGS[id as usize].dispatch(id, n),
    }
}

/// One measured recommendation: applies to GEMMs with this `(k, n)` and
/// `m_lo ≤ M ≤ m_hi`.
#[derive(Clone, Copy, Debug)]
pub struct ShapeRule {
    pub k: usize,
    pub n: usize,
    pub m_lo: usize,
    pub m_hi: usize,
    pub kernel: GemmKernel,
}

const fn r(k: usize, n: usize, m_lo: usize, m_hi: usize, kernel: GemmKernel) -> ShapeRule {
    ShapeRule { k, n, m_lo, m_hi, kernel }
}

/// Measured on the M1 Pro 14-core.
/// Rules are checked in order; the first match wins. M classes: ≤ 8 and 9–16
/// use the streaming kernel; 17–256 is the "card" class (M = 163), > 256 the
/// "ticket" class (M = 493, and the 512-row prefill chunk).
pub const RECOMMENDED: &[ShapeRule] = &[
    // ---- M ≤ 16: streaming kernel. Wide N: 32 columns per simdgroup, 4-way split-K;
    // narrow N (≤ 2560): 16 columns per simdgroup and 8-way split-K for M ≤ 8. ----
    r(2560, 12352, 1, 8, GemmKernel::Small(11)),
    r(2560, 12352, 9, 16, GemmKernel::Small(3)),
    r(2560, 10240, 1, 8, GemmKernel::Small(11)),
    r(2560, 10240, 9, 16, GemmKernel::Small(3)),
    r(2560, 18432, 1, 8, GemmKernel::Small(11)),
    r(2560, 18432, 9, 16, GemmKernel::Small(3)),
    r(4096, 2560, 1, 8, GemmKernel::Small(23)),
    r(4096, 2560, 9, 16, GemmKernel::Small(3)),
    r(9216, 2560, 1, 8, GemmKernel::Small(23)),
    r(9216, 2560, 9, 16, GemmKernel::Small(21)),
    r(2560, 2048, 1, 8, GemmKernel::Small(23)),
    r(2560, 2048, 9, 16, GemmKernel::Small(24)),
    r(2560, 8192, 1, 8, GemmKernel::Small(11)),
    r(2560, 8192, 9, 16, GemmKernel::Small(3)),
    // ---- prefill rows: tile kernel ----
    // wide N: 64x64x32, 2x2 simdgroups (128 threads), fragment group 2, load mode 1
    r(2560, 12352, 17, usize::MAX, GemmKernel::Tile(10)),
    r(2560, 10240, 17, usize::MAX, GemmKernel::Tile(10)),
    r(2560, 18432, 17, usize::MAX, GemmKernel::Tile(10)),
    r(2560, 8192, 17, usize::MAX, GemmKernel::Tile(10)),
    // N = 2560: 32x32 tiles (480 threadgroups) for the card class, 64x32 / 32x32 up to
    // 384 rows, bigger tiles for the ticket class and the 512-row prefill chunk
    r(4096, 2560, 17, 256, GemmKernel::Tile(12)),
    r(4096, 2560, 257, 384, GemmKernel::Tile(13)),
    r(4096, 2560, 385, usize::MAX, GemmKernel::Tile(10)),
    r(9216, 2560, 17, 384, GemmKernel::Tile(12)),
    r(9216, 2560, 385, usize::MAX, GemmKernel::Tile(2)),
    // N = 2048: 64x32 tiles with 2x2 simdgroups
    r(2560, 2048, 17, usize::MAX, GemmKernel::Tile(13)),
];

/// Default choices for shapes outside the table.
pub const DEFAULT_SMALL: GemmKernel = GemmKernel::Small(3);
pub const DEFAULT_TILE: GemmKernel = GemmKernel::Tile(10);

/// The recommended kernel for `Y[M, N] = X[M, K] · Wᵀ` with the given epilogue.
pub fn recommend(m: usize, n: usize, k: usize, epilogue: u32) -> GemmKernel {
    let ok = |kernel: GemmKernel| match kernel {
        GemmKernel::Tile(id) => TILE_CFGS[id as usize].supports(k, epilogue),
        GemmKernel::Small(id) => SMALL_CFGS[id as usize].supports(m, k, epilogue),
    };
    for rule in RECOMMENDED {
        if rule.k == k && rule.n == n && m >= rule.m_lo && m <= rule.m_hi && ok(rule.kernel) {
            return rule.kernel;
        }
    }
    if m <= 16 && ok(DEFAULT_SMALL) {
        return DEFAULT_SMALL;
    }
    if ok(DEFAULT_TILE) {
        return DEFAULT_TILE;
    }
    // K % 32 == 0 is the contract; every bk = 32 config satisfies it, and
    // config 0 supports every epilogue.
    GemmKernel::Tile(0)
}

/// Recommended kernel and its dispatch geometry.
pub fn plan(m: usize, n: usize, k: usize, epilogue: u32) -> GemmDispatch {
    dispatch_for(recommend(m, n, k, epilogue), m, n, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menus_are_consistent() {
        for (id, c) in TILE_CFGS.iter().enumerate() {
            assert_eq!(c.bm % (8 * c.wm), 0, "cfg {id}");
            assert_eq!(c.bn % (8 * c.wn), 0, "cfg {id}");
            assert!(c.threads() <= 256, "cfg {id}");
            assert!(c.smem_bytes() <= 32 * 1024, "cfg {id}");
        }
        for (id, c) in SMALL_CFGS.iter().enumerate() {
            assert_eq!(c.nsg % c.ks, 0, "small cfg {id}");
            assert!(c.threads() <= 256, "small cfg {id}");
        }
    }

    /// Parses `X(id, ...)` rows of a menu block in the Metal source.
    fn metal_menu(name: &str) -> Vec<Vec<u32>> {
        let src = include_str!("../../../kernels/gemm_fast.metal");
        let start = src.find(&format!("#define {name}(X)")).unwrap_or_else(|| panic!("{name} not found"));
        let block = &src[start..];
        let block = &block[..block.find("\n\n").expect("menu ends at a blank line")];
        block
            .lines()
            .filter_map(|l| l.trim().strip_prefix("X("))
            .map(|row| {
                row.split(')').next().unwrap().split(',').map(|v| v.trim().parse::<u32>().expect("menu number")).collect()
            })
            .collect()
    }

    #[test]
    fn menus_match_the_metal_source() {
        let tiles = metal_menu("GEMM_TILE_MENU");
        assert_eq!(tiles.len(), TILE_CFGS.len(), "tile menu length");
        for (i, row) in tiles.iter().enumerate() {
            let c = TILE_CFGS[i];
            assert_eq!(row[0] as usize, i, "tile menu ids must be dense and in order");
            assert_eq!(&row[1..], &[c.bm, c.bn, c.bk, c.wm, c.wn, c.pf, c.g, c.lm], "tile cfg {i}");
        }
        let smalls = metal_menu("GEMM_SMALL_MENU");
        assert_eq!(smalls.len(), SMALL_CFGS.len(), "small menu length");
        for (i, row) in smalls.iter().enumerate() {
            let c = SMALL_CFGS[i];
            assert_eq!(row[0] as usize, i, "small menu ids must be dense and in order");
            assert_eq!(&row[1..], &[c.tm, c.tn, c.nsg, c.ks], "small cfg {i}");
        }
    }

    #[test]
    fn recommended_uses_only_engine_marked_entries() {
        // The menus mark entries the engine table uses; keep the marks honest.
        let src = include_str!("gemm_configs.rs");
        let marked = |menu: &str, id: u32| {
            let block = &src[src.find(menu).unwrap()..];
            block.lines().any(|l| l.contains(&format!("// {id:>2} engine")))
        };
        for rule in RECOMMENDED {
            match rule.kernel {
                GemmKernel::Tile(id) => assert!(marked("pub const TILE_CFGS", id), "tile {id} is not marked `engine`"),
                GemmKernel::Small(id) => assert!(marked("pub const SMALL_CFGS", id), "small {id} is not marked `engine`"),
            }
        }
    }

    #[test]
    fn plan_covers_engine_shapes() {
        for (k, n) in [(2560, 12352), (2560, 10240), (2560, 18432), (4096, 2560), (9216, 2560), (2560, 2048), (2560, 8192)] {
            for m in [1, 8, 16, 163, 300, 493, 512] {
                let d = plan(m, n, k, 0);
                assert!(d.grid.0 > 0 && d.grid.1 > 0 && d.threads > 0);
            }
        }
        assert!(matches!(recommend(1, 12352, 2560, 0), GemmKernel::Small(_)));
        assert!(matches!(recommend(493, 12352, 2560, 0), GemmKernel::Tile(_)));
        // SwiGLU never gets a WN = 4 tile or a 16-column small config.
        for m in [1, 16, 163, 493] {
            match recommend(m, 18432, 2560, 3) {
                GemmKernel::Tile(id) => assert!(TILE_CFGS[id as usize].supports_swiglu()),
                GemmKernel::Small(id) => assert!(SMALL_CFGS[id as usize].supports_swiglu()),
            }
        }
    }
}
