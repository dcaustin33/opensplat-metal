# opensplat-metal

Forward-only Python bindings for the OpenSplat Metal gaussian splatting rasterizer on Apple Silicon. No gradients — designed for rendering pre-trained 3DGS `.ply` models from Python.

## Quick Start

```bash
# Requires a venv with PyTorch (MPS-enabled). We share OpenSplat's:
source ~/Desktop/OpenSplat/.venv/bin/activate
pip install -e .
python tests/test_smoke.py
python tests/test_render.py   # renders ~/Downloads/banana.ply -> output/
```

## Architecture

```
opensplat_metal/
├── csrc/
│   ├── ext.mm               # pybind11 module (_C) — 7 forward functions + BLOCK constants
│   ├── gsplat_metal.mm      # Vendored from OpenSplat, modified (see below)
│   ├── gsplat_metal.metal   # Metal shader — vendored unmodified from OpenSplat
│   ├── bindings.h           # Forward-only C++ declarations (backward removed)
│   └── config.h             # BLOCK_X=16, BLOCK_Y=16
├── __init__.py               # Re-exports _C functions + render()
└── render.py                 # High-level render() pipeline
```

### Vendored Source Modifications (from OpenSplat)

**`gsplat_metal.mm`** (from `OpenSplat/rasterizer/gsplat-metal/gsplat_metal.mm`):
- Include path changed: `#import "../gsplat/config.h"` → `#import "config.h"`
- Metallib lookup replaced: NSBundle → `dladdr`-based (finds `.so` directory at runtime, looks for `default.metallib` there), with fallback to runtime `.metal` source compilation
- All backward functions removed (compute_sh_backward, project_gaussians_backward, nd_rasterize_backward, rasterize_backward)
- Backward pipeline state objects removed from `MetalContext` struct
- Backward kernel loading removed from `init_gsplat_metal_context()`

**`bindings.h`** (from `OpenSplat/rasterizer/gsplat-metal/bindings.h`):
- Trimmed to forward-only declarations (6 functions)

**`gsplat_metal.metal`** — copied unmodified. Backward kernels are still in the file but never loaded.

## Build System

Uses `setup.py` with `torch.utils.cpp_extension.CppExtension`:
- Metal shader compiled to `default.metallib` via `xcrun -sdk macosx metal` during `pip install`
- Custom `BuildExtension` copies `default.metallib` next to the built `.so`
- Links `-framework Metal -framework MetalKit -framework Foundation`
- **Source paths in setup.py must be relative** (setuptools rejects absolute paths)

## Python API

### `opensplat_metal.render()`

High-level pipeline that chains all rasterization steps:

```python
out_img = opensplat_metal.render(
    means,        # (N, 3) float32 MPS — positions
    quats,        # (N, 4) float32 MPS — quaternions (normalized)
    scales,       # (N, 3) float32 MPS — positive scales (already exp'd)
    sh_coeffs,    # (N, K, 3) float32 MPS — SH coefficients
    opacities,    # (N,) float32 MPS — values in [0,1] (already sigmoid'd)
    viewmat,      # (4, 4) float32 MPS — world-to-camera
    K,            # (3, 3) float32 MPS — camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    img_height, img_width,
    background=None,   # (3,) or None (defaults to black)
    glob_scale=1.0,
    clip_thresh=0.01,
    sh_degree=None,    # override SH degree (None = infer from sh_coeffs shape)
)
# Returns: (H, W, 3) float32 MPS tensor
```

### Low-level C functions (`opensplat_metal._C`)

| Function | Purpose |
|----------|---------|
| `compute_sh_forward` | SH coefficients → view-dependent RGB |
| `project_gaussians_forward` | 3D gaussians → 2D screen projection |
| `map_gaussian_to_intersects` | Projected gaussians → tile intersections |
| `get_tile_bin_edges` | Sorted intersections → tile bin ranges |
| `nd_rasterize_forward` | Tile-sorted gaussians → image (N-channel) |
| `rasterize_forward` | Same as above (3-channel variant) |
| `compute_cov2d_bounds` | 2D covariance → conics + radii |

## Critical Implementation Details

### Projection matrix
`projmat = proj @ viewmat` where `proj` is the OpenGL perspective matrix built from intrinsics (matching `OpenSplat/model.cpp:35`). Both `proj` and `viewmat` are row-major 4x4.

### SH color shift
`compute_sh_forward` returns values centered at 0. You **must** add `+ 0.5` to shift into [0,1] RGB range. This is the standard 3DGS convention. Forgetting this makes everything render black.

### SH degree slicing
When rendering at a lower SH degree than stored in the PLY (e.g., `sh_degree=0` for DC-only), the coefficient tensor must be sliced to `(N, num_bases, 3)` before passing to `compute_sh_forward`. The kernel validates this shape strictly.

### Tile binning pipeline
Must follow this exact sequence (matching `OpenSplat/rasterize_gaussians.cpp:62-65`):
1. `num_tiles_hit` from `project_gaussians_forward`
2. `cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)`
3. `map_gaussian_to_intersects(... cum_tiles_hit ...)` — **not** raw `num_tiles_hit`
4. `torch.sort(isect_ids)` → sorted IDs + indices
5. `torch.gather(gaussian_ids, 0, sort_indices)` → sorted gaussian IDs
6. `get_tile_bin_edges(num_intersects, isect_ids_sorted)` → tile bins

### Input conventions
- All tensors must be **MPS** device and **contiguous**
- Scales: already exponentiated (positive), not log-scale
- Opacities: already sigmoided [0,1], not logits
- Quaternions: should be normalized (the render function does not normalize)
- `KMP_DUPLICATE_LIB_OK` is set in `__init__.py` to avoid OpenMP conflicts

## Loading PLY Files

See `tests/test_render.py` for a complete PLY parser. Key field mappings:
- Positions: `x, y, z`
- Scales: `scale_0, scale_1, scale_2` → apply `torch.exp()`
- Rotations: `rot_0, rot_1, rot_2, rot_3` → normalize to unit quaternion
- Opacities: `opacity` → apply `torch.sigmoid()`
- SH DC: `f_dc_0, f_dc_1, f_dc_2` → shape `(N, 1, 3)` as first SH basis
- SH rest: `f_rest_0` ... `f_rest_N` → reshape to `(N, num_bases-1, 3)`, interleaved per-basis

## Testing

- `tests/test_smoke.py` — verifies extension loads, constants, SH forward, projection
- `tests/test_render.py` — end-to-end: loads `~/Downloads/banana.ply`, renders 8 orbit views, saves to `output/`
