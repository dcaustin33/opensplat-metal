# opensplat-metal

Forward-only Python bindings for the [OpenSplat](https://github.com/pierotofy/OpenSplat) Metal gaussian splatting rasterizer on Apple Silicon. Renders pre-trained 3DGS `.ply` models from Python — no gradients, no training.

## Requirements

- Apple Silicon Mac (Metal GPU)
- Python with PyTorch (MPS-enabled)
- Xcode command line tools

## Installation

```bash
# Activate a PyTorch venv (e.g., OpenSplat's)
source ~/Desktop/OpenSplat/.venv/bin/activate

pip install -e .
```

This compiles the Metal shader and C++ extension automatically.

## Usage

```python
import opensplat_metal

# High-level: render from tensors (all must be float32 on MPS)
img = opensplat_metal.render(
    means,       # (N, 3) — Gaussian centers
    quats,       # (N, 4) — unit quaternions
    scales,      # (N, 3) — positive scales (pre-exponentiated)
    sh_coeffs,   # (N, K, 3) — spherical harmonic coefficients
    opacities,   # (N,)  — values in [0, 1] (pre-sigmoided)
    viewmat,     # (4, 4) — world-to-camera transform
    K,           # (3, 3) — camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    img_height,
    img_width,
)
# Returns (H, W, 3) float32 MPS tensor
```

See `tests/test_render.py` for a complete example that loads a `.ply` file and renders 8 orbit views.

```bash
# Download a model, then:
python tests/test_render.py   # renders ~/Downloads/banana.ply → output/
python tests/test_smoke.py    # basic sanity checks
```

## PLY Field Conventions

| Field | Transform |
|-------|-----------|
| `x, y, z` | positions — use as-is |
| `scale_0/1/2` | apply `torch.exp()` |
| `rot_0/1/2/3` | normalize to unit quaternion |
| `opacity` | apply `torch.sigmoid()` |
| `f_dc_0/1/2` | SH DC component → shape `(N, 1, 3)` |
| `f_rest_*` | SH rest → reshape to `(N, num_bases-1, 3)`, interleaved per-basis |

## Architecture

```
opensplat_metal/
├── csrc/
│   ├── ext.mm              # pybind11 module
│   ├── gsplat_metal.mm     # Metal rasterizer (vendored from OpenSplat, forward-only)
│   ├── gsplat_metal.metal  # Metal shader (vendored unmodified)
│   ├── bindings.h          # Forward-only declarations
│   └── config.h            # BLOCK_X=16, BLOCK_Y=16
├── __init__.py             # Re-exports _C + render()
└── render.py               # High-level render() pipeline
```

## Low-level API

`opensplat_metal._C` exposes: `compute_sh_forward`, `project_gaussians_forward`, `map_gaussian_to_intersects`, `get_tile_bin_edges`, `nd_rasterize_forward`, `rasterize_forward`, `compute_cov2d_bounds`.

See `CLAUDE.md` for the exact tile-binning pipeline order and other implementation details.
