// decide — fast causal GQA attention with the sigmoid output gate.
//
// Derived from MLX steel attention (MIT, see kernels/steel/steel_frag.h and
// mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h): a
// threadgroup stages K and V tiles of BK cache rows, each simdgroup holds
// S = Q·Kᵀ for 8 query rows as simdgroup-matrix fragments, runs the online
// softmax on the fragments (row max/sum via two lane shuffles) and
// accumulates O += P·V in fp32 fragments. What differs from MLX:
//   * one threadgroup = one kv head × 8·RG consecutive query rows; its 4·RG
//     simdgroups are the 4 query heads sharing that kv head (× RG row groups),
//     so one staged K/V tile serves 4 heads and the causal block limit is
//     exact to 8 rows (MLX tiles 32 rows of one head).
//   * queries live in registers as 32 pre-packed half2 fragments (no Q tile).
//   * K and V tiles are copied with 16-byte vectors. The Kᵀ operand comes
//     through a permutation of the head dimension (a dot product is
//     order-independent, and Q is packed to match): either packed per MMA
//     from two 16-byte threadgroup loads (KP = 0) or, with the tile stored
//     key-pair interleaved by the loader (KP = 1), as a bare 16-byte load. The
//     V operand is one 16-byte load whose four words are four fragments'
//     operands, through a matching permutation of the output columns (undone
//     in the store).
//   * lazy rescaling: O is multiplied by the correction factor only when some
//     row's running max moved in this block.
//   * absolute positions: query row r sits at position qpos[r] and sees cache
//     rows 0..=qpos[r]; positions need not be consecutive (the pruned last
//     layer), rows past `rows` are clamped for loads and skipped for stores.
//   * the sigmoid output gate is applied in the store.
//   * finite sentinel -1e30 instead of -INFINITY (the build uses fast math).
//   * row groups run longest-first (threadgroup x reversed), so the tail of
//     the causal triangle is not the last work scheduled.
//
// Contract: buffer 0 q half [rows,16,256]; 1 kc, 2 vc
// half [.., 4, 256] (kv head j of position pos at (pos·4 + j)·256); 3 Pq float
// with the gate at column 4096 + h·256 + d, row stride ldq; 4 qpos uint
// [rows]; 5 y half [rows, 4096]; 6 AttnFastParams {ldq, rows}. Softmax over
// scores/16 in fp32; y = softmax(qkᵀ/16)·v ⊙ sigmoid(gate). P (the softmax
// weights) is rounded to fp16 once, as the PV MMA's operand.
// Requires 16-byte-aligned q/kc/vc byte offsets, an 8-byte-aligned Pq offset
// with even ldq (float2 gate reads), and 4-byte-aligned qpos/y offsets.
// Cache rows 0..=max(qpos) must have been written (finite values) before the
// dispatch: rows in (qpos[r], max(qpos)] are staged and masked for row r,
// and a NaN there would reach O through the PV MMA (0 · NaN). Rows past
// max(qpos) are never read.
//
// Dispatch: threadgroups (ceil(rows / (8·RG)), 4) of 128·RG threads,
// threadgroup memory 0 = AttnCfg::SMEM_BYTES, which the engine reads from
// the metallib through the `attn_hd256_fast_smem` probe kernel (see the
// bottom of this file) rather than mirroring the formula.

#include "common.h"
#include "steel/steel_frag.h"

using namespace steel;

struct AttnFastParams { uint ldq; uint rows; };

// 1/16 · log2(e): scores are scaled and the softmax runs in exp2.
constant float kAttnScaleLog2 = 0.0625f * 1.4426950408889634f;
constant float kAttnNegSentinel = -1e30f;

// RG row groups of 8 rows per threadgroup (4·RG simdgroups), BK cache rows per
// block, PF 1 = the next tile is fetched into registers during this block's
// MMAs, ALIAS 1 = K and V share one tile buffer (V is staged after the QK
// product, MLX style; halves the threadgroup memory, doubles the barriers),
// KP 1 = the K tile is stored key-pair interleaved so the Kᵀ operand is a bare
// 16-byte load (the packing moves into the loader, shared by the 4·RG
// simdgroups), PAD 1 = tile row strides ≡ 64 B (V, unpaired K) / 128 B
// (paired K) mod 512 B, so the 4-8 rows a fragment load touches hit disjoint
// bank ranges.
template <int RG, int BK, int PF, int ALIAS, int KP, int PAD>
struct AttnCfg {
    STEEL_CONST int NSG = 4 * RG;
    STEEL_CONST int TGP = 32 * NSG;
    STEEL_CONST int TK = BK / 8;                 // key fragments per block
    // V (and unpaired K) row stride in halves: PAD 0 = 16-byte pad, 1 = bank-spread, 2 = none.
    STEEL_CONST int LD = PAD == 1 ? 288 : (PAD == 2 ? 256 : 264);
    STEEL_CONST int LDP = PAD == 1 ? 576 : (PAD == 2 ? 512 : 520);   // paired K row stride (2 keys)
    STEEL_CONST int KT_HALVES = KP ? (BK / 2) * LDP : BK * LD;
    STEEL_CONST int VT_HALVES = BK * LD;
    STEEL_CONST int V_OFF = ALIAS ? 0 : KT_HALVES;
    STEEL_CONST int SMEM_HALVES = ALIAS ? (KT_HALVES > VT_HALVES ? KT_HALVES : VT_HALVES) : KT_HALVES + VT_HALVES;
    STEEL_CONST int SMEM_BYTES = SMEM_HALVES * 2;
    STEEL_CONST int NV4 = BK * 32 / TGP;         // 16-byte copies per thread per tile
    static_assert(BK == 16 || BK == 32 || BK == 64, "BK");
    static_assert((BK * 32) % TGP == 0 && NV4 % 2 == 0, "tile copy divisibility");
    static_assert(SMEM_BYTES <= 32768, "threadgroup memory");
    static_assert(SMEM_HALVES % 8 == 0, "the tiles are declared as uint4");
};

// One K or V tile (BK cache rows of one kv head) staged through registers in
// its natural layout. Thread t copies column (t & 31) of rows (t >> 5) +
// i·(TGP/32). Row indices past `pmax` (the last position any row of the
// threadgroup attends to) are clamped: those keys are masked for every row,
// so their contents are moot, and the read stays inside the cache.
template <int BK, int TGP, int LD4>
struct AttnTile {
    STEEL_CONST int NV4 = BK * 32 / TGP;
    uint4 reg[NV4];

    // `src` = the cache as uint4 plus the kv head's 32-uint4 offset; a
    // position's row of one head is 32 uint4 and positions are 128 apart.
    // 32-bit indices: the last row, (MAX_TOKENS − 1) · 128 + 31 = 8191 · 128 + 31, is < 2^20.
    METAL_FUNC void fetch(device const uint4* src, uint kb, uint pmax, ushort tid) {
        const uint col = tid & 31u, key0 = kb * uint(BK) + (tid >> 5);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < NV4; ++i) {
            const uint key = min(key0 + uint(i * (TGP / 32)), pmax);
            reg[i] = src[key * 128u + col];
        }
    }
    METAL_FUNC void store(threadgroup uint4* dst, ushort tid) const {
        const uint col = tid & 31u, r0 = tid >> 5;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < NV4; ++i) dst[(r0 + uint(i * (TGP / 32))) * uint(LD4) + col] = reg[i];
    }
};

// The K tile stored key-pair interleaved (KP = 1): pair row pr holds keys 2pr
// and 2pr + 1; within it, head-dim group g (64 dims = 8 fragments × 8 inner
// indices) occupies 64 words at g·128 halves, word (f/4)·32 + sm·4 + (f%4) =
// (K[2pr][64g + 8sm + f], K[2pr+1][64g + 8sm + f]). Lane (sm, sn) then reads
// its 8 Kᵀ operands of (g, key fragment kf) as two 16-byte loads at pair row
// kf·4 + sn/2, halves g·128 + sm·8 and + 64. Thread t stages column t & 31
// (dims 8c..8c+8, i.e. g = c/8, sm = c%8) of pair rows (t >> 5) + i·(TGP/32).
template <int BK, int TGP, int LDP4>
struct AttnTileP {
    STEEL_CONST int NP = BK * 16 / TGP;          // key pairs per thread
    uint4 k0[NP], k1[NP];

    METAL_FUNC void fetch(device const uint4* src, uint kb, uint pmax, ushort tid) {
        const uint col = tid & 31u, p0 = tid >> 5;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < NP; ++i) {
            const uint key = kb * uint(BK) + 2u * (p0 + uint(i * (TGP / 32)));
            k0[i] = src[min(key, pmax) * 128u + col];
            k1[i] = src[min(key + 1u, pmax) * 128u + col];
        }
    }
    METAL_FUNC void store(threadgroup uint4* dst, ushort tid) const {
        const uint col = tid & 31u, p0 = tid >> 5;
        const uint g = col >> 3, sm = col & 7u;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < NP; ++i) {
            const uint4 a = k0[i], b = k1[i];
            // Word j of a/b holds dims 2j, 2j+1 of keys 2pr / 2pr+1: interleave.
            const uint4 lo = uint4((a.x & 0xffffu) | (b.x << 16), (a.x >> 16) | (b.x & 0xffff0000u),
                                   (a.y & 0xffffu) | (b.y << 16), (a.y >> 16) | (b.y & 0xffff0000u));
            const uint4 hi = uint4((a.z & 0xffffu) | (b.z << 16), (a.z >> 16) | (b.z & 0xffff0000u),
                                   (a.w & 0xffffu) | (b.w << 16), (a.w >> 16) | (b.w & 0xffff0000u));
            threadgroup uint4* row = dst + (p0 + uint(i * (TGP / 32))) * uint(LDP4) + g * 16u + sm;
            row[0] = lo;      // f = 0..3
            row[8] = hi;      // f = 4..7 (64 halves = 8 uint4 further)
        }
    }
};

// Query fragments: for head-dim group g (64 dims) and fragment f, the MMA's
// inner index r maps to d = 64g + 8r + f. Lane (sm, sn) supplies rows sm,
// inner sn and sn + 1, i.e. Q[row][64g + 8sn + f] and Q[row][64g + 8sn + 8 + f].
METAL_FUNC void attn_load_q(device const half* qrow, short sn, thread half2* qf) {
    STEEL_PRAGMA_UNROLL
    for (int g = 0; g < 4; ++g) {
        uint4 qa[1], qb[1];
        load_halves_dev<8>(qrow + g * 64 + sn * 8, qa);
        load_halves_dev<8>(qrow + g * 64 + sn * 8 + 8, qb);
        STEEL_PRAGMA_UNROLL
        for (int f = 0; f < 8; ++f) qf[g * 8 + f] = half2(half_at(qa, f), half_at(qb, f));
    }
}

// S[kf] = Q · Kᵀ for the block's TK key fragments (fp32 fragments, two
// accumulation chains per key fragment). The Kᵀ operand of (g, f) for lane
// (sm, sn) is (K[key sn][64g + 8sm + f], K[key sn + 1][64g + 8sm + f]): with
// KP = 0 two 16-byte loads per (g, key fragment) and one pack per MMA, with
// KP = 1 two 16-byte loads and no packing.
// DIAG (lab ablations, wrong results by design): bit 1 skips the tile copies,
// bit 2 the K operand loads, bit 4 the V operand loads, bit 8 the MMAs.
template <int TK, int LD, int LDP, int KP, int DIAG = 0>
METAL_FUNC void attn_qk(threadgroup const half* Ks, thread const half2* qf, thread float2* S, short sm, short sn) {
    float2 acc[TK][2];
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < TK; ++kf) { acc[kf][0] = float2(0.0f); acc[kf][1] = float2(0.0f); }
    STEEL_PRAGMA_UNROLL
    for (int g = 0; g < 4; ++g) {
        uint4 ka[TK][1], kb[TK][1];
        STEEL_PRAGMA_UNROLL
        for (int kf = 0; kf < TK; ++kf) {
            if (DIAG & 2) {
                ka[kf][0] = uint4(uint(sm) + uint(g) + uint(kf));
                kb[kf][0] = uint4(uint(sn) + uint(g));
            } else if (KP) {
                threadgroup const half* row = Ks + (kf * 4 + sn / 2) * LDP + g * 128 + sm * 8;
                load_halves_tg<8>(row, ka[kf]);
                load_halves_tg<8>(row + 64, kb[kf]);
            } else {
                load_halves_tg<8>(Ks + (kf * 8 + sn) * LD + g * 64 + sm * 8, ka[kf]);
                load_halves_tg<8>(Ks + (kf * 8 + sn + 1) * LD + g * 64 + sm * 8, kb[kf]);
            }
        }
        STEEL_PRAGMA_UNROLL
        for (int f = 0; f < 8; ++f) {
            STEEL_PRAGMA_UNROLL
            for (int kf = 0; kf < TK; ++kf) {
                half2 b;
                if (KP) {
                    b = (f < 4) ? as_type<half2>(ka[kf][0][f & 3]) : as_type<half2>(kb[kf][0][f & 3]);
                } else {
                    b = half2(half_at(ka[kf], f), half_at(kb[kf], f));
                }
                if (DIAG & 8) {
                    acc[kf][f & 1] += float2(float(b.x) * float(qf[g * 8 + f].x), float(b.y));
                } else {
                    mma_8x8(acc[kf][f & 1], qf[g * 8 + f], b);
                }
            }
        }
    }
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < TK; ++kf) S[kf] = acc[kf][0] + acc[kf][1];
}

// O[g·4 + f] += P · V. Output fragment (g, f) column c holds head dim
// d = 32g + 8·(c/2) + 2f + (c & 1), so lane (sm, sn)'s V operands for the four
// fragments of group g are the four words of V[key sm][32g + 8·(sn/2) .. +8).
template <int TK, int LD, int DIAG = 0>
METAL_FUNC void attn_pv(threadgroup const half* Vs, thread const half2* P, thread float2* O, short sm, short sn) {
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < TK; ++kf) {
        STEEL_PRAGMA_UNROLL
        for (int g = 0; g < 8; ++g) {
            uint4 vv[1];
            if (DIAG & 4) {
                vv[0] = uint4(uint(sm) + uint(g) + uint(kf));
            } else {
                load_halves_tg<8>(Vs + (kf * 8 + sm) * LD + g * 32 + (sn / 2) * 8, vv);
            }
            STEEL_PRAGMA_UNROLL
            for (int f = 0; f < 4; ++f) {
                if (DIAG & 8) {
                    const half2 v = as_type<half2>(vv[0][f]);
                    O[g * 4 + f] += float2(float(v.x) * float(P[kf].x), float(v.y));
                } else {
                    mma_8x8(O[g * 4 + f], P[kf], as_type<half2>(vv[0][f]));
                }
            }
        }
    }
}

// Scale, causal mask (keys past this lane's row position), online softmax
// update of (m, l) and O, and P = exp2(S − m) as half2 MMA operands. Row
// reductions: the four lanes of a fragment row differ in lane bits 0 and 3.
template <int TK>
METAL_FUNC void attn_softmax(thread float2* S, thread half2* P, thread float2* O, thread float& m, thread float& l,
                             uint key0, uint prow, bool masked, short sn) {
    float mloc = kAttnNegSentinel;
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < TK; ++kf) {
        S[kf] *= kAttnScaleLog2;
        if (masked) {
            const uint key = key0 + uint(kf * 8) + uint(sn);
            if (key > prow) S[kf].x = kAttnNegSentinel;
            if (key + 1u > prow) S[kf].y = kAttnNegSentinel;
        }
        mloc = max(mloc, max(S[kf].x, S[kf].y));
    }
    mloc = max(mloc, simd_shuffle_xor(mloc, 1));
    mloc = max(mloc, simd_shuffle_xor(mloc, 8));
    const float m_new = max(m, mloc);
    const float factor = fast::exp2(m - m_new);
    float lloc = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < TK; ++kf) {
        const half2 p = half2(fast::exp2(S[kf] - m_new));
        P[kf] = p;
        // l sums the fp16-rounded weights the MMA sees, so O / l is exactly normalised.
        lloc += float(p.x) + float(p.y);
    }
    lloc += simd_shuffle_xor(lloc, 1);
    lloc += simd_shuffle_xor(lloc, 8);
    l = l * factor + lloc;
    m = m_new;
    if (simd_any(factor != 1.0f)) {
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 32; ++i) O[i] *= factor;
    }
}

template <int RG, int BK, int PF, int ALIAS, int KP, int PAD, int DIAG = 0>
METAL_FUNC void attn_fast_run(device const half* q, device const half* kc, device const half* vc,
                              device const float* Pq, device const uint* qpos, device half* y,
                              constant AttnFastParams& p, threadgroup half* smem,
                              uint2 tg, ushort sg, ushort lane)
{
    using C = AttnCfg<RG, BK, PF, ALIAS, KP, PAD>;
    constexpr int TK = C::TK, LD = C::LD, LDP = C::LDP;
    const ushort tid = sg * 32 + lane;
    const short2 fc = frag_coord(lane);
    const short sn = fc.x, sm = fc.y;

    // Longest row groups first.
    const uint ngroups = (p.rows + uint(8 * RG) - 1) / uint(8 * RG);
    const uint rb_tg = (ngroups - 1u - tg.x) * uint(8 * RG);
    const uint j = tg.y;                                   // kv head
    const uint h = j * 4u + uint(sg % 4);                  // this simdgroup's query head
    const uint nrows_tg = min(uint(8 * RG), p.rows - rb_tg);
    const uint r = rb_tg + uint(sg / 4) * 8u + uint(sm);   // this lane's row
    const uint rc = min(r, p.rows - 1u);
    const uint prow = qpos[rc];

    // Positions of the threadgroup's rows: the last block, and where masking starts.
    const uint pv = qpos[rb_tg + min(uint(lane), nrows_tg - 1u)];
    const uint pmax = simd_max(pv), pmin = simd_min(pv);
    const uint nblocks = (pmax + uint(BK)) / uint(BK);
    const uint kb_mask = (pmin + 1u) / uint(BK);

    half2 qf[32];
    attn_load_q(q + (ulong(rc) * 16u + h) * 256u, sn, qf);

    float2 O[32];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 32; ++i) O[i] = float2(0.0f);
    float m = kAttnNegSentinel, l = 0.0f;
    float2 S[TK];
    half2 P[TK];

    device const uint4* kc4 = (device const uint4*)kc + j * 32u;
    device const uint4* vc4 = (device const uint4*)vc + j * 32u;
    threadgroup half* Ks = smem;
    threadgroup half* Vs = smem + C::V_OFF;
    threadgroup uint4* Ks4 = (threadgroup uint4*)Ks;
    threadgroup uint4* Vs4 = (threadgroup uint4*)Vs;
    AttnTile<BK, C::TGP, LD / 8> vt;
    typename conditional<KP != 0, AttnTileP<BK, C::TGP, LDP / 8>, AttnTile<BK, C::TGP, LD / 8>>::type kt;

    if (ALIAS == 0) {
        if (PF == 0) {
            for (uint kb = 0; kb < nblocks; ++kb) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!(DIAG & 1)) {
                    kt.fetch(kc4, kb, pmax, tid); kt.store(Ks4, tid);
                    vt.fetch(vc4, kb, pmax, tid); vt.store(Vs4, tid);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                attn_qk<TK, LD, LDP, KP, DIAG>(Ks, qf, S, sm, sn);
                attn_softmax<TK>(S, P, O, m, l, kb * uint(BK), prow, kb >= kb_mask, sn);
                attn_pv<TK, LD, DIAG>(Vs, P, O, sm, sn);
            }
        } else {
            kt.fetch(kc4, 0, pmax, tid); kt.store(Ks4, tid);
            vt.fetch(vc4, 0, pmax, tid); vt.store(Vs4, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kb = 0; kb < nblocks; ++kb) {
                const bool more = kb + 1u < nblocks;
                if (more) { kt.fetch(kc4, kb + 1u, pmax, tid); vt.fetch(vc4, kb + 1u, pmax, tid); }
                attn_qk<TK, LD, LDP, KP>(Ks, qf, S, sm, sn);
                attn_softmax<TK>(S, P, O, m, l, kb * uint(BK), prow, kb >= kb_mask, sn);
                attn_pv<TK, LD>(Vs, P, O, sm, sn);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (more) { kt.store(Ks4, tid); vt.store(Vs4, tid); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        if (PF == 0) {
            for (uint kb = 0; kb < nblocks; ++kb) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                kt.fetch(kc4, kb, pmax, tid); kt.store(Ks4, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                attn_qk<TK, LD, LDP, KP>(Ks, qf, S, sm, sn);
                attn_softmax<TK>(S, P, O, m, l, kb * uint(BK), prow, kb >= kb_mask, sn);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                vt.fetch(vc4, kb, pmax, tid); vt.store(Vs4, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                attn_pv<TK, LD>(Vs, P, O, sm, sn);
            }
        } else {
            kt.fetch(kc4, 0, pmax, tid); kt.store(Ks4, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kb = 0; kb < nblocks; ++kb) {
                const bool more = kb + 1u < nblocks;
                vt.fetch(vc4, kb, pmax, tid);
                attn_qk<TK, LD, LDP, KP>(Ks, qf, S, sm, sn);
                attn_softmax<TK>(S, P, O, m, l, kb * uint(BK), prow, kb >= kb_mask, sn);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                vt.store(Vs4, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (more) kt.fetch(kc4, kb + 1u, pmax, tid);
                attn_pv<TK, LD>(Vs, P, O, sm, sn);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (more) kt.store(Ks4, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (r >= p.rows) return;
    const float inv_l = 1.0f / l;
    device const float* gate = Pq + ulong(r) * p.ldq + 4096u + h * 256u;
    device half* yrow = y + ulong(r) * 4096u + h * 256u;
    STEEL_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
        STEEL_PRAGMA_UNROLL
        for (int f = 0; f < 4; ++f) {
            const int d = g * 32 + (sn / 2) * 8 + 2 * f;
            const float2 gt = *((device const float2*)(gate + d));
            const float2 o = O[g * 4 + f] * inv_l * float2(sigmoid_f(gt.x), sigmoid_f(gt.y));
            *((device half2*)(yrow + d)) = half2(o);
        }
    }
}

// TGA = the max_total_threads_per_threadgroup attribute (sets the register budget).
#define ATTN_FAST_KERNEL_D(NAME, RG, BK, PF, ALIAS, KP, PAD, TGA, DIAG) \
[[kernel, max_total_threads_per_threadgroup(TGA)]] \
void NAME(device const half*  q    [[buffer(0)]], \
          device const half*  kc   [[buffer(1)]], \
          device const half*  vc   [[buffer(2)]], \
          device const float* Pq   [[buffer(3)]], \
          device const uint*  qpos [[buffer(4)]], \
          device half*        y    [[buffer(5)]], \
          constant AttnFastParams& p [[buffer(6)]], \
          threadgroup half* smem [[threadgroup(0)]], \
          uint2 tg [[threadgroup_position_in_grid]], \
          ushort sg [[simdgroup_index_in_threadgroup]], \
          ushort lane [[thread_index_in_simdgroup]]) \
{ attn_fast_run<RG, BK, PF, ALIAS, KP, PAD, DIAG>(q, kc, vc, Pq, qpos, y, p, smem, tg, sg, lane); }
#define ATTN_FAST_KERNEL_T(NAME, RG, BK, PF, ALIAS, KP, PAD, TGA) ATTN_FAST_KERNEL_D(NAME, RG, BK, PF, ALIAS, KP, PAD, TGA, 0)
#define ATTN_FAST_KERNEL(NAME, RG, BK, PF, ALIAS, KP, PAD) ATTN_FAST_KERNEL_T(NAME, RG, BK, PF, ALIAS, KP, PAD, 128 * RG)

// Lab twin with the tiles declared statically in the kernel (as uint4, for
// the 16-byte alignment the tile accesses need). The compiler emits ~23%
// slower code for this form, so
// the engine kernel keeps host-sized tiles and publishes their size through
// ATTN_FAST_SMEM_PROBE below.
#define ATTN_FAST_KERNEL_STATIC(NAME, RG, BK, PF, ALIAS, KP, PAD) \
[[kernel, max_total_threads_per_threadgroup(128 * RG)]] \
void NAME(device const half*  q    [[buffer(0)]], \
          device const half*  kc   [[buffer(1)]], \
          device const half*  vc   [[buffer(2)]], \
          device const float* Pq   [[buffer(3)]], \
          device const uint*  qpos [[buffer(4)]], \
          device half*        y    [[buffer(5)]], \
          constant AttnFastParams& p [[buffer(6)]], \
          uint2 tg [[threadgroup_position_in_grid]], \
          ushort sg [[simdgroup_index_in_threadgroup]], \
          ushort lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup uint4 smem[AttnCfg<RG, BK, PF, ALIAS, KP, PAD>::SMEM_HALVES / 8]; \
    attn_fast_run<RG, BK, PF, ALIAS, KP, PAD, 0>(q, kc, vc, Pq, qpos, y, p, (threadgroup half*)smem, tg, sg, lane); \
}

// Publishes AttnCfg<...>::SMEM_BYTES to the host as this kernel's *static*
// threadgroup memory (MTLComputePipelineState.staticThreadgroupMemoryLength),
// so the wrapper binds the right dynamic length for the kernel of the same
// configuration without mirroring the formula. The array is indexed at run
// time so it cannot be shrunk.
#define ATTN_FAST_SMEM_PROBE(NAME, RG, BK, PF, ALIAS, KP, PAD) \
[[kernel]] void NAME(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]], \
                     uint tgn [[threads_per_threadgroup]]) \
{ \
    constexpr uint n = uint(AttnCfg<RG, BK, PF, ALIAS, KP, PAD>::SMEM_HALVES / 8); \
    threadgroup uint4 tiles[n]; \
    for (uint i = tid; i < n; i += tgn) tiles[i] = uint4(i); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    out[tid] = tiles[(tid * 7u + 3u) % n].x; \
}

// Variant menu, kept for benchmarking the tile choices. The engine's pick is
// in crates/decide-engine/src/kernels/attn.rs.
//                                        name           RG BK PF ALIAS KP PAD
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a0_kp0_pad0, 1, 16, 0, 0, 0, 0)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a0_kp0_pad1, 1, 16, 0, 0, 0, 1)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a0_kp1_pad0, 1, 16, 0, 0, 1, 0)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a0_kp1_pad1, 1, 16, 0, 0, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf1_a0_kp1_pad1, 1, 16, 1, 0, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a1_kp1_pad1, 1, 16, 0, 1, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg1_bk32_pf0_a1_kp1_pad1, 1, 32, 0, 1, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg2_bk16_pf0_a0_kp1_pad1, 2, 16, 0, 0, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg2_bk32_pf0_a1_kp1_pad1, 2, 32, 0, 1, 1, 1)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf0_a0_kp0_pad2, 1, 16, 0, 0, 0, 2)
ATTN_FAST_KERNEL(attn_lab_rg1_bk32_pf0_a0_kp0_pad2, 1, 32, 0, 0, 0, 2)
ATTN_FAST_KERNEL(attn_lab_rg2_bk16_pf0_a0_kp0_pad2, 2, 16, 0, 0, 0, 2)
ATTN_FAST_KERNEL(attn_lab_rg1_bk16_pf1_a0_kp0_pad2, 1, 16, 1, 0, 0, 2)
ATTN_FAST_KERNEL_T(attn_lab_rg1_bk16_pf0_a0_kp0_pad0_t256, 1, 16, 0, 0, 0, 0, 256)
ATTN_FAST_KERNEL_STATIC(attn_lab_rg1_bk16_pf0_a0_kp0_pad0_static, 1, 16, 0, 0, 0, 0)
// Ablations of rg1_bk16_pf0_a0_kp0_pad0 (wrong results by design).
ATTN_FAST_KERNEL_D(attn_lab_abl_noload, 1, 16, 0, 0, 0, 0, 128, 1)
ATTN_FAST_KERNEL_D(attn_lab_abl_nok, 1, 16, 0, 0, 0, 0, 128, 2)
ATTN_FAST_KERNEL_D(attn_lab_abl_nov, 1, 16, 0, 0, 0, 0, 128, 4)
ATTN_FAST_KERNEL_D(attn_lab_abl_nokv, 1, 16, 0, 0, 0, 0, 128, 6)
ATTN_FAST_KERNEL_D(attn_lab_abl_nomma, 1, 16, 0, 0, 0, 0, 128, 8)
ATTN_FAST_KERNEL_D(attn_lab_abl_nokv_nomma, 1, 16, 0, 0, 0, 0, 128, 14)
ATTN_FAST_KERNEL_D(attn_lab_abl_nothing, 1, 16, 0, 0, 0, 0, 128, 15)

// The engine's kernel (the fastest variant measured on an M1 Pro):
// 128 threads (kv head × 8 rows), BK 16, separate K|V tiles with a 16-byte
// pad (16.9 KB), packed Kᵀ operand, no register prefetch. Its configuration
// is written once, here; attn_hd256_fast_smem publishes the tile size to
// crates/decide-engine/src/kernels/attn.rs.
#define ATTN_ENGINE_CFG 1, 16, 0, 0, 0, 0
// Indirection so ATTN_ENGINE_CFG expands into separate macro arguments.
#define ATTN_WITH_CFG(M, NAME, ...) M(NAME, __VA_ARGS__)
ATTN_WITH_CFG(ATTN_FAST_KERNEL, attn_hd256_fast, ATTN_ENGINE_CFG)
ATTN_WITH_CFG(ATTN_FAST_SMEM_PROBE, attn_hd256_fast_smem, ATTN_ENGINE_CFG)

#undef ATTN_WITH_CFG
#undef ATTN_ENGINE_CFG
#undef ATTN_FAST_SMEM_PROBE
#undef ATTN_FAST_KERNEL_STATIC
#undef ATTN_FAST_KERNEL
