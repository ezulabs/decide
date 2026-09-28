// decide — fast fp16 GEMM, Y = X·Wᵀ with fused epilogues.
//
// Derived from MLX steel (MIT, see kernels/steel/steel_frag.h): threadgroup-tiled
// simdgroup-matrix GEMM with transposed B. What differs from MLX:
//   * k-permuted fragments: a dot product is order-independent, so the k index
//     inside a BK chunk is permuted such that every lane reads its A (and B)
//     fragment elements as one vector from threadgroup memory instead of MLX's
//     2-byte element loads. Two load modes (LM) are provided, see TileCfg.
//   * register prefetch / double buffering of the next k-chunk (PF = 1 / 2).
//   * M tail in-kernel at 8·WM-row granularity (16 rows for the recommended
//     tiles, which interleave 8-row fragments across WM = 2 simdgroups): rows
//     past M re-read the last valid row, row fragments entirely past M skip
//     their MMAs (compile-time variants chosen per threadgroup), stores are
//     masked. No 64-row padding.
//   * threadgroup x walks the M tiles (GEMM_ORDER = 1), so the few M tiles of
//     one W stripe run back to back and W streams from DRAM once.
//   * the four epilogues written straight from the accumulator fragments,
//     including SwiGLU (gate/up pairing is a fragment pair 16 columns apart).
//   * one entry point per threadgroup size (gemm_nt_fast = 64 threads,
//     gemm_nt_fast_tg32/128/256), because max_total_threads_per_threadgroup
//     sets the register budget (256 cost 7% at 64 threads).
//   * `gemm_nt_small`: a threadgroup-memory-free streaming kernel for M ≤ 32
//     (weights read once, 128-byte-coalesced, x from cache; optional split-K
//     across the simdgroups of a threadgroup).
//
// Contract: buffer 0 x half [M,K] row stride ldx; 1 w half [N,K]
// row stride ldw; 2 y (float*, reinterpreted per epilogue); 3 GemmParams
// {M, N, K, ldx, ldw, ldy, swiglu_block}. Function constant 0 = epilogue
// (0 store f32, 1 store f16, 2 y += acc f32, 3 SwiGLU → f16 [M, N/2]).
// Function constant 1 = tile config id (menu below; default 0), 2 = grid order
// (1: threadgroup x walks M tiles, default; 0: x walks N tiles, MLX order).
// Requires N % 32 == 0, K % BK == 0, ldx % 8 == 0, ldw % 8 == 0, ldy even
// (float2/half2 stores), swiglu_block == 16. M arbitrary (≥ 1).
//
// Dispatch (see crates/decide-metal/src/gemm_configs.rs):
//   gemm_nt_fast*: threadgroups (ceil(M/BM), N/BN) [order 1] of WM·WN·32 threads,
//                  threadgroup memory 0 = TileCfg::SMEM_BYTES.
//   gemm_nt_small: threadgroups (ceil(N / (NSG/KS · TN · 8)), 1) of NSG·32 threads,
//                  threadgroup memory 0 = (KS-1)·(NSG/KS)·32·TM·TN·8 bytes (0 if KS = 1).

#include "common.h"
#include "steel/steel_frag.h"

using namespace steel;

constant uint GEMM_EPI [[function_constant(0)]];
constant uint GEMM_CFG_FC [[function_constant(1)]];
constant uint GEMM_ORDER_FC [[function_constant(2)]];
constant bool GEMM_CFG_SET = is_function_constant_defined(GEMM_CFG_FC);
constant bool GEMM_ORDER_SET = is_function_constant_defined(GEMM_ORDER_FC);
constant uint GEMM_CFG = GEMM_CFG_SET ? GEMM_CFG_FC : 0u;
constant uint GEMM_ORDER = GEMM_ORDER_SET ? GEMM_ORDER_FC : 1u;

struct GemmParams { uint M; uint N; uint K; uint ldx; uint ldw; uint ldy; uint swiglu_block; };

template <int TM, int TN>
struct Acc {
    float2 v[TM][TN];
    METAL_FUNC void clear() {
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (int j = 0; j < TN; ++j) v[i][j] = float2(0.0f);
        }
    }
};

// ---------------------------------------------------------------------------
// Fused epilogue from accumulator fragments.
// This lane owns rows row0 + i·RS and columns col0 + j·CS (and +1), i < TM, j < TN.
// RS = 8·WM and CS = 8·WN for the interleaved simdgroup layout. SwiGLU pairs a
// gate fragment with the fragment 16 columns to its right, j + 16/CS, so it
// needs CS ∈ {8, 16} (WN ∈ {1, 2}) and 32-column-aligned simdgroup tiles.
// ---------------------------------------------------------------------------
template <int TM, int TN, int RS, int CS>
METAL_FUNC void store_acc(thread const Acc<TM, TN>& acc, device float* y,
                          constant GemmParams& p, uint row0, uint col0)
{
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
        const uint row = row0 + uint(i * RS);
        if (row >= p.M) continue;
        const ulong rbase = ulong(row) * p.ldy;
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
            const uint col = col0 + uint(j * CS);
            if (col >= p.N) continue;
            const float2 a = acc.v[i][j];
            if (GEMM_EPI == 3) {
                // Gate fragments are the ones at column offsets 0..15 of each
                // 32-column block; the lane part of `col` (< 16) never changes that.
                if (((j * CS) % 32) < 16) {
                    const int jp = (j + 16 / CS < TN) ? (j + 16 / CS) : (TN - 1);
                    const float2 u = acc.v[i][jp];
                    const float2 o = float2(silu_f(a.x) * u.x, silu_f(a.y) * u.y);
                    const uint oc = (col / 32) * 16 + (col % 16);
                    *((device half2*)((device half*)y + rbase + oc)) = half2(o);
                }
            } else if (GEMM_EPI == 0) {
                *((device float2*)(y + rbase + col)) = a;
            } else if (GEMM_EPI == 1) {
                *((device half2*)((device half*)y + rbase + col)) = half2(a);
            } else {
                device float2* q = (device float2*)(y + rbase + col);
                *q = *q + a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tile kernel geometry.
//   BM×BN output tile, BK-wide k-chunks, WM×WN simdgroups (interleaved 8-row /
//   8-column fragments), PF prefetch mode (0 plain, 1 register prefetch,
//   2 double-buffered threadgroup memory), G fragments per fragment group,
//   LM load mode. Within a group (8·G consecutive k), fragment f and k index r:
//     LM 0: k = G·r + f. Lane (fm, fn) reads A[m][G·fn .. +2G) and
//           W[n][G·fm .. +G) for n = fn, fn+1; every MMA operand is packed from
//           two registers.
//     LM 1: k = G·(r & ~1) + 2f + (r & 1). The A read is the same vector but
//           each operand is one 32-bit word (no packing); B reads its two rows
//           element-wise (2-byte loads, MLX style).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int PF, int G, int LM>
struct TileCfg {
    STEEL_CONST int TGP = WM * WN * 32;
    STEEL_CONST int TM = BM / (8 * WM);
    STEEL_CONST int TN = BN / (8 * WN);
    STEEL_CONST int F = BK / 8;                 // fragments per k-chunk
    STEEL_CONST int LD = BK + 8;                // tgmem row stride (halves); 16-byte pad
    STEEL_CONST int A_HALVES = BM * LD;
    STEEL_CONST int B_HALVES = BN * LD;
    STEEL_CONST int BUF_HALVES = A_HALVES + B_HALVES;
    STEEL_CONST int NBUF = (PF == 2) ? 2 : 1;
    STEEL_CONST int SMEM_BYTES = BUF_HALVES * NBUF * 2;
    // Loaders: each thread copies NR contiguous halves of one row per pass.
    STEEL_CONST int A_NR = (BM * BK / TGP < BK) ? (BM * BK / TGP) : BK;
    STEEL_CONST int A_TCOLS = BK / A_NR;
    STEEL_CONST int A_TROWS = TGP / A_TCOLS;
    STEEL_CONST int A_PASSES = BM / A_TROWS;
    STEEL_CONST int A_VEC = A_NR / 8;
    STEEL_CONST int B_NR = (BN * BK / TGP < BK) ? (BN * BK / TGP) : BK;
    STEEL_CONST int B_TCOLS = BK / B_NR;
    STEEL_CONST int B_TROWS = TGP / B_TCOLS;
    STEEL_CONST int B_PASSES = BN / B_TROWS;
    STEEL_CONST int B_VEC = B_NR / 8;
    static_assert(BM % (8 * WM) == 0 && BN % (8 * WN) == 0, "tile / simdgroup geometry");
    static_assert(BK == 16 || BK == 32 || BK == 64, "BK");
    static_assert(G == 2 || G == 4 || G == 8, "fragment group");
    static_assert(BK % (8 * G) == 0, "fragment groups must tile the chunk");
    static_assert(LM == 0 || LM == 1, "load mode");
    static_assert(A_NR % 8 == 0 && B_NR % 8 == 0, "loader rows must split into 16-byte vectors");
    static_assert((BM * BK) % TGP == 0 && (BN * BK) % TGP == 0, "loader divisibility");
    static_assert(BM % A_TROWS == 0 && BN % B_TROWS == 0, "loader passes");
};

// Device → registers → threadgroup copy of one operand tile.
template <int PASSES, int VEC, int STEP>
struct VecLoader {
    device const uint4* src[PASSES];
    threadgroup uint4* dst[PASSES];
    uint4 reg[PASSES][VEC];

    METAL_FUNC void fetch(uint kc) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < PASSES; ++q) {
            STEEL_PRAGMA_UNROLL
            for (int v = 0; v < VEC; ++v) reg[q][v] = src[q][kc * STEP + v];
        }
    }
    METAL_FUNC void store(int buf) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < PASSES; ++q) {
            STEEL_PRAGMA_UNROLL
            for (int v = 0; v < VEC; ++v) dst[q][buf + v] = reg[q][v];
        }
    }
    METAL_FUNC void copy(uint kc) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < PASSES; ++q) {
            STEEL_PRAGMA_UNROLL
            for (int v = 0; v < VEC; ++v) dst[q][v] = src[q][kc * STEP + v];
        }
    }
};

// One BK-wide chunk of MMAs from threadgroup memory for the first TMV row
// fragments (TMV = TM for full tiles). All loop bounds are compile-time so the
// fragment vectors stay in registers.
template <int TM, int TN, int F, int G, int LD, int WM, int WN, int TMV, int LM>
METAL_FUNC void mma_chunk(threadgroup const half* a_rd, threadgroup const half* b_rd,
                          thread Acc<TM, TN>& acc)
{
    STEEL_PRAGMA_UNROLL
    for (int g = 0; g < F / G; ++g) {
        const int kg = g * 8 * G;
        uint4 av[TMV][(2 * G + 7) / 8];
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < TMV; ++i) load_halves_tg<2 * G>(a_rd + i * (8 * WM) * LD + kg, av[i]);
        if (LM == 0) {
            uint4 bv[TN][2][(G + 7) / 8];
            STEEL_PRAGMA_UNROLL
            for (int j = 0; j < TN; ++j) {
                load_halves_tg<G>(b_rd + (j * (8 * WN)) * LD + kg, bv[j][0]);
                load_halves_tg<G>(b_rd + (j * (8 * WN) + 1) * LD + kg, bv[j][1]);
            }
            STEEL_PRAGMA_UNROLL
            for (int f = 0; f < G; ++f) {
                STEEL_PRAGMA_UNROLL
                for (int i = 0; i < TMV; ++i) {
                    const half2 a = half2(half_at(av[i], f), half_at(av[i], G + f));
                    STEEL_PRAGMA_UNROLL
                    for (int j = 0; j < TN; ++j) {
                        const half2 b = half2(half_at(bv[j][0], f), half_at(bv[j][1], f));
                        mma_8x8(acc.v[i][j], a, b);
                    }
                }
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (int f = 0; f < G; ++f) {
                half2 b[TN];
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < TN; ++j) {
                    b[j] = half2(b_rd[(j * (8 * WN)) * LD + kg + 2 * f],
                                 b_rd[(j * (8 * WN) + 1) * LD + kg + 2 * f]);
                }
                STEEL_PRAGMA_UNROLL
                for (int i = 0; i < TMV; ++i) {
                    const half2 a = as_type<half2>(av[i][f >> 2][f & 3]);
                    STEEL_PRAGMA_UNROLL
                    for (int j = 0; j < TN; ++j) mma_8x8(acc.v[i][j], a, b[j]);
                }
            }
        }
    }
}

// The k loop for one threadgroup with TMV valid row fragments per simdgroup.
template <int BM, int BN, int BK, int WM, int WN, int PF, int G, int LM, int TMV>
METAL_FUNC void gemm_tile_loop(
    thread VecLoader<TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::A_PASSES, TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::A_VEC, BK / 8>& la,
    thread VecLoader<TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::B_PASSES, TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::B_VEC, BK / 8>& lb,
    threadgroup const half* a_rd, threadgroup const half* b_rd,
    thread Acc<TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::TM, TileCfg<BM, BN, BK, WM, WN, PF, G, LM>::TN>& acc,
    const uint iters)
{
    using C = TileCfg<BM, BN, BK, WM, WN, PF, G, LM>;
    constexpr int TM = C::TM, TN = C::TN, F = C::F, LD = C::LD;
    constexpr int BUF_U4 = C::BUF_HALVES / 8;

    if (PF == 0) {
        for (uint kc = 0; kc < iters; ++kc) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            la.copy(kc);
            lb.copy(kc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_chunk<TM, TN, F, G, LD, WM, WN, TMV, LM>(a_rd, b_rd, acc);
        }
    } else if (PF == 1) {
        la.fetch(0);
        lb.fetch(0);
        for (uint kc = 0; kc < iters; ++kc) {
            la.store(0);
            lb.store(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kc + 1 < iters) { la.fetch(kc + 1); lb.fetch(kc + 1); }
            mma_chunk<TM, TN, F, G, LD, WM, WN, TMV, LM>(a_rd, b_rd, acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        la.fetch(0);
        lb.fetch(0);
        la.store(0);
        lb.store(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kc = 0; kc < iters; ++kc) {
            const int cur = int(kc & 1u) * C::BUF_HALVES;
            const int nxt = int((kc + 1u) & 1u) * BUF_U4;
            const bool more = kc + 1 < iters;
            if (more) { la.fetch(kc + 1); lb.fetch(kc + 1); }
            mma_chunk<TM, TN, F, G, LD, WM, WN, TMV, LM>(a_rd + cur, b_rd + cur, acc);
            if (more) { la.store(nxt); lb.store(nxt); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int PF, int G, int LM>
METAL_FUNC void gemm_tile_run(device const half* x, device const half* w, device float* y,
                              constant GemmParams& p, threadgroup half* smem,
                              uint2 tg, ushort sg, ushort lane)
{
    using C = TileCfg<BM, BN, BK, WM, WN, PF, G, LM>;
    constexpr int TM = C::TM, TN = C::TN, LD = C::LD;

    const uint mt = (GEMM_ORDER != 0u) ? tg.x : tg.y;
    const uint nt = (GEMM_ORDER != 0u) ? tg.y : tg.x;
    const uint m0 = mt * BM, n0 = nt * BN;
    if (m0 >= p.M || n0 >= p.N) return;
    const int rows_valid = int(p.M - m0);          // ≥ 1
    const int cols_valid = int(p.N - n0);          // ≥ 32
    const ushort tid = sg * 32 + lane;

    VecLoader<C::A_PASSES, C::A_VEC, BK / 8> la;
    {
        const int bi = tid / C::A_TCOLS, bj = (tid % C::A_TCOLS) * C::A_NR;
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < C::A_PASSES; ++q) {
            const int r = bi + q * C::A_TROWS;
            const int rc = min(r, rows_valid - 1);
            la.src[q] = (device const uint4*)(x + ulong(m0 + uint(rc)) * p.ldx + bj);
            la.dst[q] = (threadgroup uint4*)(smem + r * LD + bj);
        }
    }
    VecLoader<C::B_PASSES, C::B_VEC, BK / 8> lb;
    {
        const int bi = tid / C::B_TCOLS, bj = (tid % C::B_TCOLS) * C::B_NR;
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < C::B_PASSES; ++q) {
            const int r = bi + q * C::B_TROWS;
            const int rc = min(r, cols_valid - 1);
            lb.src[q] = (device const uint4*)(w + ulong(n0 + uint(rc)) * p.ldw + bj);
            lb.dst[q] = (threadgroup uint4*)(smem + C::A_HALVES + r * LD + bj);
        }
    }

    const short2 fc = frag_coord(lane);
    const short sn = fc.x, sm = fc.y;
    const short tm = 8 * (sg / WN), tn = 8 * (sg % WN);
    threadgroup const half* a_rd = smem + (tm + sm) * LD + G * sn;
    threadgroup const half* b_rd = smem + C::A_HALVES + (tn + sn) * LD
        + ((LM == 0) ? (G * sm) : (G * (sm & ~1) + (sm & 1)));

    Acc<TM, TN> acc;
    acc.clear();
    const uint iters = p.K / BK;

    // Row fragments needed by the simdgroup with the most valid rows (tm = 0):
    // ceil(rows_valid / (8·WM)). Uniform over the threadgroup, so the barriers
    // inside the chosen loop are uniform too. Fragments ≥ tmv hold only rows ≥ M.
    const int tmv = min(TM, (rows_valid + 8 * WM - 1) / (8 * WM));
#define GEMM_TMV_CASE(V) \
    if (TM >= V && tmv == V) { gemm_tile_loop<BM, BN, BK, WM, WN, PF, G, LM, (V < TM ? V : TM)>(la, lb, a_rd, b_rd, acc, iters); } else
    GEMM_TMV_CASE(1) GEMM_TMV_CASE(2) GEMM_TMV_CASE(3) GEMM_TMV_CASE(4)
    GEMM_TMV_CASE(5) GEMM_TMV_CASE(6) GEMM_TMV_CASE(7)
    { gemm_tile_loop<BM, BN, BK, WM, WN, PF, G, LM, TM>(la, lb, a_rd, b_rd, acc, iters); }
#undef GEMM_TMV_CASE

    store_acc<TM, TN, 8 * WM, 8 * WN>(acc, y, p, m0 + uint(tm + sm), n0 + uint(tn + sn));
}

// Instantiated only for the entry point whose threadgroup size matches.
template <bool ENABLED, int BM, int BN, int BK, int WM, int WN, int PF, int G, int LM>
struct TileEntry {
    static METAL_FUNC void run(device const half* x, device const half* w, device float* y,
                               constant GemmParams& p, threadgroup half* smem,
                               uint2 tg, ushort sg, ushort lane) {
        gemm_tile_run<BM, BN, BK, WM, WN, PF, G, LM>(x, w, y, p, smem, tg, sg, lane);
    }
};
template <int BM, int BN, int BK, int WM, int WN, int PF, int G, int LM>
struct TileEntry<false, BM, BN, BK, WM, WN, PF, G, LM> {
    static METAL_FUNC void run(device const half*, device const half*, device float*,
                               constant GemmParams&, threadgroup half*, uint2, ushort, ushort) {}
};

// Tile config menu. Keep in sync with TILE_CFGS in
// crates/decide-metal/src/gemm_configs.rs (same ids, same order; a unit test
// there parses this block). `engine` entries are referenced by the engine's
// RECOMMENDED table; `lab` entries exist for gemm_lab sweeps.
//                 id  BM   BN  BK WM WN PF  G LM
#define GEMM_TILE_MENU(X) \
    X( 0,  64,  32, 32, 2, 1, 0, 4, 0) /* lab */    \
    X( 1,  64,  32, 32, 2, 1, 0, 2, 0) /* lab */    \
    X( 2,  64,  32, 32, 2, 1, 0, 4, 1) /* engine */ \
    X( 3,  64,  32, 32, 2, 1, 0, 2, 1) /* lab */    \
    X( 4,  64,  32, 32, 2, 1, 1, 4, 0) /* lab */    \
    X( 5,  64,  32, 32, 2, 1, 1, 2, 1) /* lab */    \
    X( 6,  64,  32, 32, 2, 1, 2, 2, 1) /* lab */    \
    X( 7,  64,  32, 64, 2, 1, 0, 4, 1) /* lab */    \
    X( 8,  64,  32, 64, 2, 1, 0, 2, 1) /* lab */    \
    X( 9,  64,  64, 32, 2, 2, 0, 4, 0) /* lab */    \
    X(10,  64,  64, 32, 2, 2, 0, 2, 1) /* engine */ \
    X(11,  64,  64, 32, 2, 2, 1, 2, 1) /* lab */    \
    X(12,  32,  32, 32, 2, 2, 0, 2, 1) /* engine */ \
    X(13,  64,  32, 32, 2, 2, 0, 2, 1) /* engine */ \
    X(14,  64,  32, 32, 4, 1, 0, 2, 1) /* lab */    \
    X(15,  32,  64, 32, 2, 2, 0, 2, 1) /* lab */    \
    X(16,  64,  32, 16, 2, 1, 0, 2, 1) /* lab */    \
    X(17,  64,  32, 16, 2, 1, 0, 2, 0) /* lab */    \
    X(18, 128,  32, 32, 4, 1, 0, 2, 1) /* lab */    \
    X(19, 128,  64, 32, 4, 2, 0, 2, 1) /* lab */    \
    X(20,  64,  64, 32, 1, 2, 0, 2, 1) /* lab */    \
    X(21,  32,  32, 32, 1, 1, 0, 2, 1) /* lab */    \
    X(22,  64,  32, 32, 2, 1, 2, 4, 0) /* lab */    \
    X(23,  32,  64, 32, 1, 2, 0, 2, 1) /* lab */    \
    X(24,  64,  64, 32, 2, 1, 0, 2, 1) /* lab */    \
    X(25,  64,  64, 32, 2, 1, 0, 4, 0) /* lab */    \
    X(26,  64,  32, 64, 2, 1, 0, 4, 0) /* lab */    \
    X(27,  32,  32, 32, 2, 2, 0, 4, 0) /* lab */   

#define GEMM_TILE_CASE(ID, BM, BN, BK, WM, WN, PF, G, LM) \
    if (GEMM_CFG == uint(ID)) { TileEntry<(WM * WN * 32 == kTGP), BM, BN, BK, WM, WN, PF, G, LM>::run(x, w, y, p, smem, tg, sg, lane); return; }

#define GEMM_TILE_KERNEL(NAME, TGP) \
[[kernel, max_total_threads_per_threadgroup(TGP)]] \
void NAME(device const half* x [[buffer(0)]], \
          device const half* w [[buffer(1)]], \
          device float* y [[buffer(2)]], \
          constant GemmParams& p [[buffer(3)]], \
          threadgroup half* smem [[threadgroup(0)]], \
          uint2 tg [[threadgroup_position_in_grid]], \
          ushort sg [[simdgroup_index_in_threadgroup]], \
          ushort lane [[thread_index_in_simdgroup]]) \
{ \
    constexpr int kTGP = TGP; \
    GEMM_TILE_MENU(GEMM_TILE_CASE) \
}

GEMM_TILE_KERNEL(gemm_nt_fast, 64)
GEMM_TILE_KERNEL(gemm_nt_fast_tg32, 32)
GEMM_TILE_KERNEL(gemm_nt_fast_tg128, 128)
GEMM_TILE_KERNEL(gemm_nt_fast_tg256, 256)

#undef GEMM_TILE_KERNEL
#undef GEMM_TILE_CASE

// ---------------------------------------------------------------------------
// Small-M kernel (M ≤ 8·TM): no threadgroup staging. Each simdgroup owns TN·8
// output columns and streams W once from device memory; x (tiny) comes from
// cache. Fragments are k-permuted exactly like the tile kernel, so a lane's
// reads are contiguous: A 2F halves at F·fn, B F halves at F·fm per row.
// KS > 1 splits K across the KS simdgroups of a column group and reduces
// through threadgroup memory (more simdgroups in flight for narrow N).
// ---------------------------------------------------------------------------
template <int TM, int TN, int F>
METAL_FUNC void small_chunk(device const half* x, device const half* w, constant GemmParams& p,
                            uint k0, uint n0, short sm, short sn, thread Acc<TM, TN>& acc)
{
    uint4 av[TM][(2 * F + 7) / 8];
    uint4 bv[TN][2][(F + 7) / 8];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
        const uint r = min(uint(i * 8 + sm), p.M - 1);
        load_halves_dev<2 * F>(x + ulong(r) * p.ldx + k0 + F * sn, av[i]);
    }
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int jj = 0; jj < 2; ++jj) {
            const uint n = min(n0 + uint(j * 8 + sn + jj), p.N - 1);
            load_halves_dev<F>(w + ulong(n) * p.ldw + k0 + F * sm, bv[j][jj]);
        }
    }
    STEEL_PRAGMA_UNROLL
    for (int f = 0; f < F; ++f) {
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < TM; ++i) {
            const half2 a = half2(half_at(av[i], f), half_at(av[i], F + f));
            STEEL_PRAGMA_UNROLL
            for (int j = 0; j < TN; ++j) {
                const half2 b = half2(half_at(bv[j][0], f), half_at(bv[j][1], f));
                mma_8x8(acc.v[i][j], a, b);
            }
        }
    }
}

template <int TM, int TN, int NSG, int KS>
METAL_FUNC void gemm_small_run(device const half* x, device const half* w, device float* y,
                               constant GemmParams& p, threadgroup float2* red,
                               uint2 tg, ushort sg, ushort lane)
{
    static_assert(NSG % KS == 0, "split-K groups");
    constexpr int CG = NSG / KS;
    const int cg = sg / KS, kp = sg % KS;
    const uint n0 = (tg.x * CG + uint(cg)) * uint(TN * 8);
    const bool active = n0 < p.N;
    const short2 fc = frag_coord(lane);
    const short sn = fc.x, sm = fc.y;

    Acc<TM, TN> acc;
    acc.clear();
    if (active) {
        const uint kpart = p.K / KS;                   // host: kpart % 32 == 0
        const uint kbeg = uint(kp) * kpart, kend = kbeg + kpart;
        uint k0 = kbeg;
        for (; k0 + 64 <= kend; k0 += 64) small_chunk<TM, TN, 8>(x, w, p, k0, n0, sm, sn, acc);
        if (k0 < kend) small_chunk<TM, TN, 4>(x, w, p, k0, n0, sm, sn, acc);
    }

    if (KS > 1) {
        constexpr int PER_SG = 32 * TM * TN;
        if (kp > 0) {
            threadgroup float2* dst = red + ((kp - 1) * CG + cg) * PER_SG + lane * (TM * TN);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < TM; ++i) {
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < TN; ++j) dst[i * TN + j] = acc.v[i][j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kp > 0 || !active) return;
        for (int q = 1; q < KS; ++q) {
            threadgroup const float2* src = red + ((q - 1) * CG + cg) * PER_SG + lane * (TM * TN);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < TM; ++i) {
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < TN; ++j) acc.v[i][j] += src[i * TN + j];
            }
        }
    } else if (!active) {
        return;
    }
    store_acc<TM, TN, 8, 8>(acc, y, p, uint(sm), n0 + uint(sn));
}

// Small kernel menu. Keep in sync with SMALL_CFGS in gemm_configs.rs
// (`engine` / `lab` as above).
//                  id TM TN NSG KS
#define GEMM_SMALL_MENU(X) \
    X( 0, 2, 4, 1, 1) /* lab */    \
    X( 1, 2, 4, 2, 1) /* lab */    \
    X( 2, 2, 4, 4, 1) /* lab */    \
    X( 3, 2, 4, 4, 4) /* engine */ \
    X( 4, 2, 4, 2, 2) /* lab */    \
    X( 5, 2, 4, 8, 4) /* lab */    \
    X( 6, 2, 2, 2, 1) /* lab */    \
    X( 7, 2, 2, 4, 1) /* lab */    \
    X( 8, 1, 4, 1, 1) /* lab */    \
    X( 9, 1, 4, 2, 1) /* lab */    \
    X(10, 1, 4, 4, 1) /* lab */    \
    X(11, 1, 4, 4, 4) /* engine */ \
    X(12, 1, 4, 8, 4) /* lab */    \
    X(13, 1, 2, 4, 1) /* lab */    \
    X(14, 4, 4, 2, 1) /* lab */    \
    X(15, 4, 4, 4, 4) /* lab */    \
    X(16, 2, 8, 2, 1) /* lab */    \
    X(17, 2, 8, 4, 4) /* lab */    \
    X(18, 1, 8, 2, 1) /* lab */    \
    X(19, 1, 8, 4, 4) /* lab */    \
    X(20, 1, 4, 8, 8) /* lab */    \
    X(21, 2, 4, 8, 8) /* engine */ \
    X(22, 1, 2, 4, 4) /* lab */    \
    X(23, 1, 2, 8, 8) /* engine */ \
    X(24, 2, 2, 4, 4) /* engine */ \
    X(25, 2, 2, 8, 8) /* lab */    \
    X(26, 1, 2, 8, 4) /* lab */    \
    X(27, 2, 2, 8, 4) /* lab */   

[[kernel, max_total_threads_per_threadgroup(256)]]
void gemm_nt_small(device const half* x [[buffer(0)]],
                   device const half* w [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant GemmParams& p [[buffer(3)]],
                   threadgroup float2* red [[threadgroup(0)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]])
{
#define GEMM_SMALL_CASE(ID, TM, TN, NSG, KS) \
    if (GEMM_CFG == uint(ID)) { gemm_small_run<TM, TN, NSG, KS>(x, w, y, p, red, tg, sg, lane); return; }
    GEMM_SMALL_MENU(GEMM_SMALL_CASE)
#undef GEMM_SMALL_CASE
}
