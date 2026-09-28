#include "common.h"

struct GNormParams { uint ldz; };

// One simdgroup per (value head, row). Grid: threads (32, 32, T); threadgroup (32, 1, 1).
kernel void gated_rmsnorm_silu(
    device const float* o [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device half*        y [[buffer(3)]],
    constant GNormParams& p [[buffer(4)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint h = gid.y, t = gid.z;
    device const float* os = o + (ulong(t) * 32 + h) * 128 + lane * 4;
    float x[4], ss = 0.0f;
    for (int i = 0; i < 4; ++i) { x[i] = os[i]; ss += x[i] * x[i]; }
    const float inv = rsqrt(simd_sum(ss) / 128.0f + 1e-6f);
    for (int i = 0; i < 4; ++i) {
        const uint d = lane * 4 + i;
        const float zz = z[ulong(t) * p.ldz + h * 128 + d];
        y[ulong(t) * 4096 + h * 128 + d] = half(x[i] * inv * w[d] * silu_f(zz));
    }
}
