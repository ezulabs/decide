// Gated DeltaNet recurrence, fast path. Drop-in for `gated_delta`:
// same buffers, same DeltaParams, same semantics as
// `cpu_ref::gated_delta`:
//
//   per token t, value head h (key head h/2):
//     g = exp(-exp(A_log[h]) * softplus(a[t] + dt_bias[h]))   beta = sigmoid(b[t])
//     S = S * g;  kv = S k;  delta = (v - kv) * beta;  S = S + k (x) delta;  o = S q
//
// Design:
//   * A simdgroup owns 16 value columns of one head and lane l holds
//     S[h][dv][4l..4l+4) for each of them (64 floats of state in registers).
//     k_t and q_t are then one coalesced float4 load per lane and never touch
//     threadgroup memory; the two dot products per column are finished with
//     a select-butterfly over the 32 lanes that costs one shuffle per column
//     per dot (not five). No threadgroup memory, no barriers.
//   * The decay is folded into a running scalar: S = c * S~. Per token and
//     element this is 3 FMAs (kv dot, rank-1 update, o dot) instead of 4;
//     c is renormalised into S~ when it drops below 2^-60, so the arithmetic
//     stays scale-invariant (no overflow/underflow) and the result matches the
//     reference to fp32 rounding.
//   * g and beta are computed once per (token, head): lane l computes them for
//     token t0 + l, 32 tokens at a time, and each token broadcasts its pair.
//     k, q, v of token t+1 and a, b of the next 32-token block are loaded one
//     step ahead.
//   * Every register array is indexed with literal constants only (the
//     DELTA_FOR* macros expand the loops): the Metal front end does not
//     unroll loops, and a loop-indexed array ends up in private memory.
//
// Dispatch: threadgroups (8, 32, 1) of (32, 1, 1) threads; threadgroup (x, h)
// owns columns [16 x, 16 x + 16) of value head h. Buffers 0-10 are `gated_delta`'s
// contract; state_in may alias state_out (each lane reads its state before
// the token loop and writes it after). Chunk-continuation dispatches that
// share a state buffer must stay on one serial encoder (the engine's buffers
// are untracked).
#include "common.h"

struct DeltaParams { uint T; uint ldab; uint write_state; };

// Below this the running scale c is folded back into the state.
constant float DELTA_C_MIN = 8.673617e-19f; // 2^-60

inline float delta_softplus(float x) {
    return x > 20.0f ? x : precise::log(1.0f + precise::exp(x));
}

#define DELTA_FOR8(X)  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define DELTA_FOR16(X) DELTA_FOR8(X) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
#define DELTA_FOR32(X) DELTA_FOR16(X) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) \
    X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)

// ============================================================================
// gated_delta_fast: transposed sequential layout.
// ============================================================================

// R: columns per simdgroup (8, 16 or 32). NSG: simdgroups per threadgroup.
// PIPE: the o-butterfly of token t is deferred and interleaved with the
// kv-butterfly of token t + 1 (it is off the recurrence's critical path;
// measured slower). TAG is unused by the code: it only makes otherwise
// identical instantiations distinct, so the lab can instantiate the engine's
// configuration under its own name.
template <int R, int NSG, int TAG = 0, int PIPE = 0>
[[kernel]] void delta_fast_kernel(
    device const float4* q         [[buffer(0)]],
    device const float4* k         [[buffer(1)]],
    device const float*  v         [[buffer(2)]],
    device const float*  a         [[buffer(3)]],
    device const float*  b         [[buffer(4)]],
    device const float*  a_log     [[buffer(5)]],
    device const float*  dt_bias   [[buffer(6)]],
    device const float4* state_in  [[buffer(7)]],
    device float4*       state_out [[buffer(8)]],
    device float*        o         [[buffer(9)]],
    constant DeltaParams& p        [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    static_assert(R == 8 || R == 16 || R == 32, "R columns per simdgroup");
    constexpr int LOG2R = R == 8 ? 3 : R == 16 ? 4 : 5;
    constexpr int SHIFT = 5 - LOG2R;            // lane >> SHIFT = the lane's column

    const uint h = tgid.y, hk = h >> 1;
    const uint col0 = (tgid.x * NSG + sg) * R;  // first column of this simdgroup
    const uint mycol = col0 + (lane >> SHIFT);  // the column this lane finishes
    const uint T = p.T;
    const uint ldab = p.ldab;

    // State: s[r] = S[h][col0 + r][4 lane .. 4 lane + 4). Coalesced rows.
    float4 s[32];
#define DELTA_LOAD_S(r) if ((r) < R) s[r] = state_in[(h * 128 + col0 + (r)) * 32 + lane];
    DELTA_FOR32(DELTA_LOAD_S)

    const float A = precise::exp(a_log[h]);
    const float bias = dt_bias[h];

    // Gate inputs for the next 32-token block: lane l holds token blk + l.
    float av = 0.0f, bv = 0.0f;
    if (lane < T) { av = a[lane * ldab + h]; bv = b[lane * ldab + h]; }
    float gv = 0.0f, betav = 0.0f;

    // Token 0's inputs.
    float4 kt = float4(0.0f), qt = float4(0.0f);
    float vt = 0.0f;
    if (T > 0) {
        kt = k[hk * 32 + lane];
        qt = q[hk * 32 + lane];
        vt = v[h * 128 + mycol];
    }

    float c = 1.0f;
    float pr[32];
    float po[32];            // PIPE: previous token's o partials
    float cprev = 1.0f;      // PIPE: its scale
#define DELTA_PO_INIT(r) if ((r) < R) po[r] = 0.0f;
    DELTA_FOR32(DELTA_PO_INIT)
    for (uint t = 0; t < T; ++t) {
        // Prefetch token t + 1.
        float4 nk = kt, nq = qt;
        float nv = vt;
        if (t + 1 < T) {
            nk = k[((t + 1) * 16 + hk) * 32 + lane];
            nq = q[((t + 1) * 16 + hk) * 32 + lane];
            nv = v[((t + 1) * 32 + h) * 128 + mycol];
        }
        // Every 32 tokens: gates for this block from the prefetched a/b,
        // then prefetch the next block's a/b.
        if ((t & 31) == 0) {
            gv = precise::exp(-A * delta_softplus(av + bias));
            betav = 1.0f / (1.0f + precise::exp(-bv));
            const uint tn = t + 32 + lane;
            if (tn < T) { av = a[tn * ldab + h]; bv = b[tn * ldab + h]; }
        }
        const float g    = simd_broadcast(gv, ushort(t & 31));
        const float beta = simd_broadcast(betav, ushort(t & 31));

        // kv partials: pr[r] = s[r] . kt
#define DELTA_KV(r) if ((r) < R) pr[r] = dot(s[r], kt);
        DELTA_FOR32(DELTA_KV)
        // Select-butterfly: after it, lane l holds column (l >> SHIFT)'s sum.
#define DELTA_BF_STEP(i)                                                             \
        if ((i) < bf_half) {                                                         \
            const float send = bf_b ? bf_p[i] : bf_p[(i) + bf_half];                 \
            const float keep = bf_b ? bf_p[(i) + bf_half] : bf_p[i];                 \
            bf_p[i] = keep + simd_shuffle_xor(send, bf_d);                           \
        }
#define DELTA_BF_LEVEL(P, d)                                                         \
        {                                                                            \
            thread float* bf_p = P;                                                  \
            constexpr int bf_half = (R * (d)) / 32;                                  \
            constexpr ushort bf_d = (d);                                             \
            if (bf_half >= 1) {                                                      \
                const bool bf_b = (lane & bf_d) != 0;                                \
                DELTA_FOR16(DELTA_BF_STEP)                                           \
            } else {                                                                 \
                bf_p[0] += simd_shuffle_xor(bf_p[0], bf_d);                          \
            }                                                                        \
        }
#define DELTA_BUTTERFLY(P) DELTA_BF_LEVEL(P, 16) DELTA_BF_LEVEL(P, 8) DELTA_BF_LEVEL(P, 4) DELTA_BF_LEVEL(P, 2) DELTA_BF_LEVEL(P, 1)
        // PIPE: the previous token's o-butterfly runs interleaved with this
        // token's kv-butterfly (two independent chains).
#define DELTA_BUTTERFLY2(P, Q) DELTA_BF_LEVEL(P, 16) DELTA_BF_LEVEL(Q, 16) DELTA_BF_LEVEL(P, 8) DELTA_BF_LEVEL(Q, 8) \
        DELTA_BF_LEVEL(P, 4) DELTA_BF_LEVEL(Q, 4) DELTA_BF_LEVEL(P, 2) DELTA_BF_LEVEL(Q, 2) DELTA_BF_LEVEL(P, 1) DELTA_BF_LEVEL(Q, 1)
        if (PIPE && t > 0) {
            DELTA_BUTTERFLY2(pr, po)
            if ((lane & ((1u << SHIFT) - 1u)) == 0) o[((t - 1) * 32 + h) * 128 + mycol] = po[0] * cprev;
        } else {
            DELTA_BUTTERFLY(pr)
        }
        const float kv = pr[0];

        float cg = c * g;                            // scale of the decayed state
        const float delta = (vt - cg * kv) * beta;
        if (cg < DELTA_C_MIN) {                      // fold the scale back in (rare)
#define DELTA_RESCALE(r) if ((r) < R) s[r] = (s[r] * c) * g;
            DELTA_FOR32(DELTA_RESCALE)
            cg = 1.0f;
        }
        const float ds = delta / cg;

        // S~[r] += kt * ds[r];  o partials pr[r] = S~[r] . qt
#define DELTA_UPD(r)                                                                 \
        if ((r) < R) {                                                               \
            const float dsr = simd_broadcast(ds, ushort((r) << SHIFT));              \
            s[r] = fma(kt, dsr, s[r]);                                               \
            pr[r] = dot(s[r], qt);                                                   \
        }
        DELTA_FOR32(DELTA_UPD)
        if (PIPE) {
#define DELTA_PO_COPY(r) if ((r) < R) po[r] = pr[r];
            DELTA_FOR32(DELTA_PO_COPY)
            cprev = cg;
        } else {
            DELTA_BUTTERFLY(pr)
            if ((lane & ((1u << SHIFT) - 1u)) == 0) o[(t * 32 + h) * 128 + mycol] = pr[0] * cg;
        }

        c = cg;
        kt = nk; qt = nq; vt = nv;
    }
    if (PIPE && T > 0) {   // flush the last token's o
        DELTA_BUTTERFLY(po)
        if ((lane & ((1u << SHIFT) - 1u)) == 0) o[((T - 1) * 32 + h) * 128 + mycol] = po[0] * cprev;
    }

    if (p.write_state) {
#define DELTA_STORE_S(r) if ((r) < R) state_out[(h * 128 + col0 + (r)) * 32 + lane] = s[r] * c;
        DELTA_FOR32(DELTA_STORE_S)
    }
#undef DELTA_LOAD_S
#undef DELTA_KV
#undef DELTA_BF_STEP
#undef DELTA_BF_LEVEL
#undef DELTA_BUTTERFLY
#undef DELTA_BUTTERFLY2
#undef DELTA_PO_INIT
#undef DELTA_PO_COPY
#undef DELTA_RESCALE
#undef DELTA_UPD
#undef DELTA_STORE_S
}

#define DELTA_FAST_INSTANTIATE(name, R, NSG, TAG)                                    \
    template [[host_name(name)]] [[kernel]] void delta_fast_kernel<R, NSG, TAG, 0>(  \
        device const float4*, device const float4*, device const float*,             \
        device const float*, device const float*, device const float*,               \
        device const float*, device const float4*, device float4*, device float*,    \
        constant DeltaParams&, uint2, uint, uint);
#define DELTA_FAST_INSTANTIATE_PIPE(name, R, NSG, TAG)                               \
    template [[host_name(name)]] [[kernel]] void delta_fast_kernel<R, NSG, TAG, 1>(  \
        device const float4*, device const float4*, device const float*,             \
        device const float*, device const float*, device const float*,               \
        device const float*, device const float4*, device float4*, device float*,    \
        constant DeltaParams&, uint2, uint, uint);

// The engine's kernel: 16 columns per simdgroup, one simdgroup per threadgroup.
DELTA_FAST_INSTANTIATE("gated_delta_fast", 16, 1, 0)
