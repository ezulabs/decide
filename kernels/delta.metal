#include "common.h"

struct DeltaParams { uint T; uint ldab; uint write_state; };

// Grid: threads (32, 128, 32) = (lane, dv, value head); threadgroup (32, 4, 1).
kernel void gated_delta(
    device const float* q         [[buffer(0)]],
    device const float* k         [[buffer(1)]],
    device const float* v         [[buffer(2)]],
    device const float* a         [[buffer(3)]],
    device const float* b         [[buffer(4)]],
    device const float* a_log     [[buffer(5)]],
    device const float* dt_bias   [[buffer(6)]],
    device const float* state_in  [[buffer(7)]],
    device float*       state_out [[buffer(8)]],
    device float*       o         [[buffer(9)]],
    constant DeltaParams& p       [[buffer(10)]],
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint dv = gid.y, h = gid.z, hk = h / 2;
    const float A = exp(a_log[h]);
    const float bias = dt_bias[h];
    float s[4];
    device const float* si = state_in + (ulong(h) * 128 + dv) * 128 + lane * 4;
    for (int i = 0; i < 4; ++i) s[i] = si[i];
    for (uint t = 0; t < p.T; ++t) {
        const float g = exp(-A * softplus_f(a[ulong(t) * p.ldab + h] + bias));
        const float beta = sigmoid_f(b[ulong(t) * p.ldab + h]);
        device const float* kt = k + (ulong(t) * 16 + hk) * 128 + lane * 4;
        device const float* qt = q + (ulong(t) * 16 + hk) * 128 + lane * 4;
        float kv = 0.0f;
        for (int i = 0; i < 4; ++i) { s[i] *= g; kv += s[i] * kt[i]; }
        kv = simd_sum(kv);
        const float delta = (v[(ulong(t) * 32 + h) * 128 + dv] - kv) * beta;
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) { s[i] += kt[i] * delta; out += s[i] * qt[i]; }
        out = simd_sum(out);
        if (lane == 0) o[(ulong(t) * 32 + h) * 128 + dv] = out;
    }
    if (p.write_state) {
        device float* so = state_out + (ulong(h) * 128 + dv) * 128 + lane * 4;
        for (int i = 0; i < 4; ++i) so[i] = s[i];
    }
}
