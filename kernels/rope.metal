#include "common.h"

struct QRopeParams  { uint ldq; };
struct KVRopeParams { uint ldkv; uint kcol; uint vcol; uint pos0; };

// Zero-centred RMSNorm over 256 dims then RoPE on dims [0,64): pairs (i, i+32).
inline float norm_rope(float x, threadgroup float* xs, threadgroup float* scratch, float wv, uint pos,
                       device const float* inv_freq, uint tid, uint lane, uint sg) {
    const float ss = tg_sum(x * x, scratch, lane, sg, 8);
    x = x * rsqrt(ss / 256.0f + 1e-6f) * (1.0f + wv);
    xs[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        const uint i = tid % 32;
        const float ang = float(pos) * inv_freq[i];
        const float c = precise::cos(ang), s = precise::sin(ang);
        const float a = xs[i], b = xs[i + 32];
        x = tid < 32 ? a * c - b * s : b * c + a * s;
    }
    return x;
}

// Grid: threadgroups (16 heads, R rows) × 256 threads.
kernel void q_norm_rope(
    device const float* Pq       [[buffer(0)]],
    device const float* qw       [[buffer(1)]],
    device const float* inv_freq [[buffer(2)]],
    device const uint*  qpos     [[buffer(3)]],
    device half*        qo       [[buffer(4)]],
    constant QRopeParams& p      [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    threadgroup float xs[256];
    const uint h = tg.x, r = tg.y;
    const float x = Pq[ulong(r) * p.ldq + h * 256 + tid];
    qo[(ulong(r) * 16 + h) * 256 + tid] = half(norm_rope(x, xs, scratch, qw[tid], qpos[r], inv_freq, tid, lane, sg));
}

// Grid: threadgroups (8 slots, R rows) × 256 threads; slots 0–3 k heads (norm+rope), 4–7 v heads (copy).
kernel void kv_norm_rope(
    device const float* Pkv      [[buffer(0)]],
    device const float* kw       [[buffer(1)]],
    device const float* inv_freq [[buffer(2)]],
    device half*        kc       [[buffer(3)]],
    device half*        vc       [[buffer(4)]],
    constant KVRopeParams& p     [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    threadgroup float xs[256];
    const uint slot = tg.x, r = tg.y, pos = p.pos0 + r;
    device const float* row = Pkv + ulong(r) * p.ldkv;
    if (slot >= 4) {
        const uint j = slot - 4;
        vc[(ulong(pos) * 4 + j) * 256 + tid] = half(row[p.vcol + j * 256 + tid]);
        return;
    }
    const float x = row[p.kcol + slot * 256 + tid];
    kc[(ulong(pos) * 4 + slot) * 256 + tid] = half(norm_rope(x, xs, scratch, kw[tid], pos, inv_freq, tid, lane, sg));
}
