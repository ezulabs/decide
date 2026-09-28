// Simdgroup-matrix fragment helpers, derived from MLX steel
// (mlx/backend/metal/kernels/steel/gemm/mma.h).
//
// MIT License
//
// Copyright © 2023-2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// What is kept from MLX: the lane → element mapping of an 8x8 simdgroup matrix
// (`frag_coord`, MLX's `BaseMMAFrag::get_coord`) and the fragment-level
// multiply-accumulate through `thread_elements()`. decide's GEMM feeds the
// fragments from 16-byte vector reads in a k-permuted order (see
// kernels/gemm_fast.metal) instead of MLX's element-wise loads.

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

namespace steel {

// Every lane of a simdgroup_matrix<T, 8, 8> holds two elements:
// (row fm, col fn) and (row fm, col fn + 1). Returns short2{fn, fm}.
// The layout is the same for the A, B and C operands of an MMA.
METAL_FUNC short2 frag_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2{fn, fm};
}

// c += a · b for one 8x8 fragment triple, fp32 accumulation.
// `a`/`b`/`c` are this lane's two elements of the respective fragments.
METAL_FUNC void mma_8x8(thread float2& c, half2 a, half2 b) {
    simdgroup_half8x8 A, B;
    simdgroup_float8x8 C, D;
    reinterpret_cast<thread half2&>(A.thread_elements()) = a;
    reinterpret_cast<thread half2&>(B.thread_elements()) = b;
    reinterpret_cast<thread float2&>(C.thread_elements()) = c;
    simdgroup_multiply_accumulate(D, A, B, C);
    c = reinterpret_cast<thread float2&>(D.thread_elements());
}

// The idx-th half of a run of 16-byte vectors held in registers.
// Only used with compile-time `idx` (inside fully unrolled loops).
METAL_FUNC half half_at(const thread uint4* v, int idx) {
    return as_type<half2>(v[idx >> 3][(idx >> 1) & 3])[idx & 1];
}

// Reads NH halves (NH ∈ {2, 4, 8, 16}) into 16-byte register vectors.
template <int NH>
METAL_FUNC void load_halves_tg(threadgroup const half* src, thread uint4* dst) {
    if (NH == 16) {
        threadgroup const uint4* s = (threadgroup const uint4*)src;
        dst[0] = s[0];
        dst[1] = s[1];
    } else if (NH == 8) {
        dst[0] = *((threadgroup const uint4*)src);
    } else if (NH == 4) {
        dst[0].xy = *((threadgroup const uint2*)src);
    } else {
        dst[0].x = *((threadgroup const uint*)src);
    }
}

template <int NH>
METAL_FUNC void load_halves_dev(device const half* src, thread uint4* dst) {
    if (NH == 16) {
        device const uint4* s = (device const uint4*)src;
        dst[0] = s[0];
        dst[1] = s[1];
    } else if (NH == 8) {
        dst[0] = *((device const uint4*)src);
    } else if (NH == 4) {
        dst[0].xy = *((device const uint2*)src);
    } else {
        dst[0].x = *((device const uint*)src);
    }
}

} // namespace steel
