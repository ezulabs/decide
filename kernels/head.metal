#include "common.h"

// final_head's row width, fixed so that it always fits hn[] below.
constant constexpr uint HIDDEN = 2560;

struct HeadParams { float eps; };

// out[i] = x[rowmap[i]] (fp32 rows). Grid: (H, rows).
kernel void gather_rows_f32(
    device const float* x      [[buffer(0)]],
    device const uint*  rowmap [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      H      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= H) return;
    out[ulong(gid.y) * H + gid.x] = x[ulong(rowmap[gid.y]) * H + gid.x];
}

// Same for fp16 rows (the pruned layer gathers normed inputs).
kernel void gather_rows_f16(
    device const half* x [[buffer(0)]], device const uint* rowmap [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint& H [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= H) return;
    out[ulong(gid.y) * H + gid.x] = x[ulong(rowmap[gid.y]) * H + gid.x];
}

// Final zero-centred norm of each answer row, then logits against its ≤62 candidate
// embedding rows (bf16); a larger ncand is clamped to 62. One threadgroup (256 threads) per row.
kernel void final_head(
    device const float*  x      [[buffer(0)]],   // [rows, HIDDEN]
    device const float*  w      [[buffer(1)]],   // final norm [HIDDEN]
    device const ushort* table  [[buffer(2)]],   // bf16 [V, HIDDEN]
    device const uint*   cand   [[buffer(3)]],   // [rows, 62]
    device const uint*   ncand  [[buffer(4)]],   // [rows]
    device float*        logits [[buffer(5)]],   // [rows, 62]
    constant HeadParams& p      [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    threadgroup float hn[HIDDEN];
    device const float* xr = x + ulong(r) * HIDDEN;
    float ss = 0.0f;
    for (uint i = tid; i < HIDDEN; i += 256) { const float v = xr[i]; ss += v * v; }
    const float inv = rsqrt(tg_sum(ss, scratch, lane, sg, 8) / float(HIDDEN) + p.eps);
    for (uint i = tid; i < HIDDEN; i += 256) hn[i] = xr[i] * inv * (1.0f + w[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = min(ncand[r], 62u);
    for (uint c = sg; c < n; c += 8) {
        device const ushort* er = table + ulong(cand[r * 62 + c]) * HIDDEN;
        float acc = 0.0f;
        for (uint i = lane; i < HIDDEN; i += 32) acc += hn[i] * bf16_to_float(er[i]);
        acc = simd_sum(acc);
        if (lane == 0) logits[r * 62 + c] = acc;
    }
}
