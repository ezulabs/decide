#pragma once
#include <metal_stdlib>
using namespace metal;

inline float bf16_to_float(ushort b) { return as_type<float>(uint(b) << 16); }
// silu_f and sigmoid_f use fast-math exp on purpose; the kernel tolerances cover its error.
inline float silu_f(float x)     { return x / (1.0f + exp(-x)); }
inline float sigmoid_f(float x)  { return 1.0f / (1.0f + exp(-x)); }
// Precise: softplus feeds the recurrence's decay gate, whose error compounds over hundreds of steps.
inline float softplus_f(float x) { return x > 20.0f ? x : precise::log(1.0f + precise::exp(x)); }

// Sum of v over the threadgroup; every thread gets the total.
// Every thread of the threadgroup must call it, in uniform control flow: it
// contains threadgroup barriers. lane, sg and nsg are the caller's
// [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]] and
// [[simdgroups_per_threadgroup]]; nsg counts a partial last simdgroup, so it
// is ceil(threads / 32), at most 32.
// `scratch` must hold 32 floats; it is reusable after the call returns.
inline float tg_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < nsg ? scratch[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) scratch[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
