#include "common.h"

struct AnePartialParams { uint h; uint ld_x; uint ld_p; float inv_scale; };

// x[r] += partial[r] * inv_scale: the Neural Engine's share of the MLP's
// down_proj. `partial` is fp16 [rows, ld_p] as the ANE wrote
// it, pre-scaled by `1 / inv_scale` (its weights' w_up carries the scale, so
// the fp16 output sits well above the ANE's absolute error floor); `x` is the
// fp32 residual stream [rows, ld_x]. Grid: (h, rows).
kernel void ane_add_partial(
    device float*      x [[buffer(0)]],
    device const half* partial [[buffer(1)]],
    constant AnePartialParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.h) return;
    x[ulong(gid.y) * p.ld_x + gid.x] += float(partial[ulong(gid.y) * p.ld_p + gid.x]) * p.inv_scale;
}
