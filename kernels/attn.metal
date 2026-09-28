#include "common.h"

struct AttnParams { uint ldq; };

// Grid: threads (32, 16, R); threadgroup (32, 1, 1). Lane owns dims [8·lane, 8·lane+8).
kernel void attn_hd256(
    device const half*  q    [[buffer(0)]],
    device const half*  kc   [[buffer(1)]],
    device const half*  vc   [[buffer(2)]],
    device const float* Pq   [[buffer(3)]],
    device const uint*  qpos [[buffer(4)]],
    device half*        y    [[buffer(5)]],
    constant AttnParams& p   [[buffer(6)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint h = gid.y, r = gid.z, kvh = h / 4, npos = qpos[r] + 1;
    float qv[8], acc[8];
    for (int i = 0; i < 8; ++i) { qv[i] = float(q[(ulong(r) * 16 + h) * 256 + lane * 8 + i]) * 0.0625f; acc[i] = 0.0f; }
    // Finite sentinel, not -INFINITY: the build uses fast (finite-only) math, where infinities are UB by contract.
    float m = -1e30f, l = 0.0f;
    for (uint j = 0; j < npos; ++j) {
        device const half* kr = kc + (ulong(j) * 4 + kvh) * 256 + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), pj = exp(s - m_new);
        l = l * corr + pj;
        device const half* vr = vc + (ulong(j) * 4 + kvh) * 256 + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + pj * float(vr[i]);
        m = m_new;
    }
    for (int i = 0; i < 8; ++i) {
        const uint d = lane * 8 + i;
        const float gate = sigmoid_f(Pq[ulong(r) * p.ldq + 4096 + h * 256 + d]);
        y[ulong(r) * 4096 + h * 256 + d] = half(acc[i] / l * gate);
    }
}
