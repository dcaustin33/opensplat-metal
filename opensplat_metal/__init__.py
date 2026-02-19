import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from opensplat_metal._C import (
    BLOCK_X,
    BLOCK_Y,
    compute_cov2d_bounds,
    compute_sh_forward,
    project_gaussians_forward,
    map_gaussian_to_intersects,
    get_tile_bin_edges,
    rasterize_forward,
    nd_rasterize_forward,
)
from opensplat_metal.render import render

__all__ = [
    "BLOCK_X",
    "BLOCK_Y",
    "compute_cov2d_bounds",
    "compute_sh_forward",
    "project_gaussians_forward",
    "map_gaussian_to_intersects",
    "get_tile_bin_edges",
    "rasterize_forward",
    "nd_rasterize_forward",
    "render",
]
