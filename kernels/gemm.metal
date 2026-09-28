#include "common.h"
#include <metal_simdgroup_matrix>

constant uint GEMM_EPI [[function_constant(0)]];   // 0 f32 · 1 f16 · 2 add f32 · 3 swiglu f16

struct GemmParams { uint M; uint N; uint K; uint ldx; uint ldw; uint ldy; uint swiglu_block; };

#define BM 32
#define BN 32
#define BK 32

// Y = X·Wᵀ, 128 threads = 4 simdgroups (2×2), each a 16×16 sub-tile of 2×2 8×8 fragments.
kernel void gemm_nt(
    device const half*   x [[buffer(0)]],
    device const half*   w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half  xs[BM * BK];
    threadgroup half  ws[BN * BK];
    threadgroup float ys[BM * BN];
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;

    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint i = tid; i < BM * BK; i += 128) {
            const uint r = i / BK, c = i % BK, gm = m0 + r;
            xs[i] = gm < p.M ? x[ulong(gm) * p.ldx + k0 + c] : half(0.0h);
        }
        for (uint i = tid; i < BN * BK; i += 128) {
            const uint r = i / BK, c = i % BK;
            ws[i] = w[ulong(n0 + r) * p.ldw + k0 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm + i * 8) * BK + kk, BK);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], ws + (sn + j * 8) * BK + kk, BK, ulong2(0, 0), true);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], ys + (sm + i * 8) * BN + sn + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (GEMM_EPI == 3) {
        const uint blk = p.swiglu_block, half_n = BN / 2;
        device half* yh = (device half*)y;
        for (uint i = tid; i < BM * half_n; i += 128) {
            const uint r = i / half_n, c = i % half_n, gm = m0 + r;
            if (gm >= p.M) continue;
            const uint b = c / blk, wi = c % blk;
            const float g = ys[r * BN + b * 2 * blk + wi];
            const float u = ys[r * BN + b * 2 * blk + blk + wi];
            yh[ulong(gm) * p.ldy + n0 / 2 + c] = half(silu_f(g) * u);
        }
        return;
    }
    for (uint i = tid; i < BM * BN; i += 128) {
        const uint r = i / BN, c = i % BN, gm = m0 + r;
        if (gm >= p.M) continue;
        const ulong o = ulong(gm) * p.ldy + n0 + c;
        if (GEMM_EPI == 0)      y[o] = ys[i];
        else if (GEMM_EPI == 1) ((device half*)y)[o] = half(ys[i]);
        else                    y[o] += ys[i];
    }
}
