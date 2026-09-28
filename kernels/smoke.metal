#include "common.h"

constant uint SMOKE_ADD [[function_constant(0)]];

kernel void smoke_add(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] += float(SMOKE_ADD);
}

// Two tg_sum calls through one scratch: each thread writes the thread count
// and the sum of thread indices to out[2*tid] and out[2*tid + 1].
kernel void smoke_tg_sum(device float* out [[buffer(0)]],
                         uint tid  [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float count = tg_sum(1.0f, scratch, lane, sg, nsg);
    float index_sum = tg_sum(float(tid), scratch, lane, sg, nsg);
    out[2 * tid] = count;
    out[2 * tid + 1] = index_sum;
}
