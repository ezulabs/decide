#include "common.h"

constant uint NORM_ADD_ONE [[function_constant(0)]];   // 1: zero-centred weight (1 + w)

struct RowNormParams { uint H; uint ld_in; uint ld_out; float eps; };

// y[r] = x[rowmap[r]] · rsqrt(mean(x²) + eps) · w'  (fp32 in, fp16 out). One threadgroup per row.
kernel void rmsnorm_rows(
    device const float* x      [[buffer(0)]],
    device const float* w      [[buffer(1)]],
    device half*        y      [[buffer(2)]],
    device const uint*  rowmap [[buffer(3)]],
    constant RowNormParams& p  [[buffer(4)]],
    uint r    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    device const float* xr = x + ulong(rowmap[r]) * p.ld_in;
    float ss = 0.0f;
    for (uint i = tid; i < p.H; i += tpg) { float v = xr[i]; ss += v * v; }
    const float inv = rsqrt(tg_sum(ss, scratch, lane, sg, (tpg + 31) / 32) / float(p.H) + p.eps);
    device half* yr = y + ulong(r) * p.ld_out;
    for (uint i = tid; i < p.H; i += tpg) {
        const float wv = NORM_ADD_ONE ? 1.0f + w[i] : w[i];
        yr[i] = half(xr[i] * inv * wv);
    }
}
