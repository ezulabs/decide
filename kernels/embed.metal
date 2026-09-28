#include "common.h"

// out[t, c] = float(table[ids[t], c]); table is bf16 [V, H]. Grid: (H, T).
kernel void embed_gather_bf16(
    device const uint*   ids   [[buffer(0)]],
    device const ushort* table [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       H     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= H) return;
    out[ulong(gid.y) * H + gid.x] = bf16_to_float(table[ulong(ids[gid.y]) * H + gid.x]);
}
