#include "common.h"

struct ConvParams { uint T; uint ldp; };

kernel void conv_silu_l2(
    device const float* P    [[buffer(0)]],
    device const float* prev [[buffer(1)]],
    device const float* w    [[buffer(2)]],
    device float*       q    [[buffer(3)]],
    device float*       k    [[buffer(4)]],
    device float*       v    [[buffer(5)]],
    constant ConvParams& p   [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    const uint slot = tg.x, t = tg.y;
    const uint c = (slot < 16 ? slot * 128 : slot < 32 ? 2048 + (slot - 16) * 128 : 4096 + (slot - 32) * 128) + tid;
    float acc = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        const int s = int(t) - 3 + int(j);
        const float xin = s >= 0 ? P[ulong(s) * p.ldp + c] : prev[uint(3 + s) * 8192 + c];
        acc += w[c * 4 + j] * xin;
    }
    const float yv = silu_f(acc);
    if (slot >= 32) { v[(ulong(t) * 32 + (slot - 32)) * 128 + tid] = yv; return; }   // uniform per threadgroup
    const float inv = rsqrt(tg_sum(yv * yv, scratch, lane, sg, 4) + 1e-6f);
    if (slot < 16) q[(ulong(t) * 16 + slot) * 128 + tid] = yv * inv * 0.08838834764831845f;   // 128^-1/2
    else           k[(ulong(t) * 16 + (slot - 16)) * 128 + tid] = yv * inv;
}

// prev_out[i] = the raw qkv row at chunk position s = T − 3 + i (i = 0..3): P[s]
// if s ≥ 0, else prev_in[3 + s]. Each thread owns one channel and reads its
// three sources before writing, so prev_out may be prev_in. Grid: threads (8192).
kernel void conv_state_update(
    device const float* P        [[buffer(0)]],
    device const float* prev_in  [[buffer(1)]],
    device float*       prev_out [[buffer(2)]],
    constant ConvParams& p       [[buffer(3)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= 8192) return;
    float r[3];
    for (int i = 0; i < 3; ++i) {
        const int s = int(p.T) - 3 + i;
        r[i] = s >= 0 ? P[ulong(s) * p.ldp + c] : prev_in[uint(3 + s) * 8192 + c];
    }
    for (int i = 0; i < 3; ++i) prev_out[uint(i) * 8192 + c] = r[i];
}
