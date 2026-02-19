import math
import torch
from opensplat_metal._C import (
    BLOCK_X,
    BLOCK_Y,
    compute_sh_forward,
    project_gaussians_forward,
    map_gaussian_to_intersects,
    get_tile_bin_edges,
    nd_rasterize_forward,
)


def _projection_matrix(fx, fy, cx, cy, img_w, img_h, z_near=0.001, z_far=1000.0, device="mps"):
    """Build the OpenGL-style projection matrix matching OpenSplat's model.cpp:35."""
    fov_x = 2.0 * math.atan(img_w / (2.0 * fx))
    fov_y = 2.0 * math.atan(img_h / (2.0 * fy))

    t = z_near * math.tan(0.5 * fov_y)
    b = -t
    r = z_near * math.tan(0.5 * fov_x)
    l = -r

    return torch.tensor([
        [2.0 * z_near / (r - l), 0.0, (r + l) / (r - l), 0.0],
        [0.0, 2.0 * z_near / (t - b), (t + b) / (t - b), 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_far - z_near), -1.0 * z_far * z_near / (z_far - z_near)],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32, device=device)


def render(
    means,        # (N, 3) positions
    quats,        # (N, 4) quaternions
    scales,       # (N, 3) positive scales (already exponentiated)
    sh_coeffs,    # (N, K, 3) SH coefficients
    opacities,    # (N,) values in [0,1] (already sigmoided)
    viewmat,      # (4, 4) world-to-camera
    K,            # (3, 3) camera intrinsics
    img_height,
    img_width,
    background=None,
    glob_scale=1.0,
    clip_thresh=0.01,
    sh_degree=None,
):
    """Render gaussians to an image using the Metal rasterizer.

    Returns:
        torch.Tensor: (H, W, 3) rendered image.
    """
    device = means.device
    N = means.shape[0]

    if background is None:
        background = torch.zeros(3, dtype=torch.float32, device=device)
    else:
        background = background.to(dtype=torch.float32, device=device)

    fx = K[0, 0].item()
    fy = K[1, 1].item()
    cx = K[0, 2].item()
    cy = K[1, 2].item()

    # Determine SH degree from coefficients shape
    num_coeffs = sh_coeffs.shape[1]
    if sh_degree is not None:
        degree = sh_degree
    else:
        # Infer from num_coeffs: 1->0, 4->1, 9->2, 16->3, 25->4
        if num_coeffs >= 25:
            degree = 4
        elif num_coeffs >= 16:
            degree = 3
        elif num_coeffs >= 9:
            degree = 2
        elif num_coeffs >= 4:
            degree = 1
        else:
            degree = 0

    # Camera position = inverse of viewmat translation
    # viewmat is world-to-camera, so camera_pos = -R^T @ t
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam_pos = -R.T @ t  # (3,)

    # View directions: from camera to each gaussian, normalized
    viewdirs = means - cam_pos.unsqueeze(0)  # (N, 3)
    viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
    viewdirs = viewdirs.contiguous()

    # 1. SH -> RGB colors
    sh_coeffs_c = sh_coeffs.contiguous()
    colors = compute_sh_forward(N, degree, degree, viewdirs, sh_coeffs_c)  # (N, 3)

    # Clamp colors to valid range
    colors = colors.clamp(min=0.0)

    # 2. Build projection matrix: projmat = proj @ viewmat
    proj = _projection_matrix(fx, fy, cx, cy, img_width, img_height, device=device)
    projmat = (proj @ viewmat).contiguous()

    viewmat_c = viewmat.contiguous()
    means_c = means.contiguous()
    scales_c = scales.contiguous()
    quats_c = quats.contiguous()

    tile_bounds = (
        (img_width + BLOCK_X - 1) // BLOCK_X,
        (img_height + BLOCK_Y - 1) // BLOCK_Y,
        1,
    )

    # 3. Project gaussians
    cov3d, xys, depths, radii, conics, num_tiles_hit = project_gaussians_forward(
        N, means_c, scales_c, glob_scale, quats_c,
        viewmat_c, projmat,
        fx, fy, cx, cy,
        img_height, img_width,
        tile_bounds, clip_thresh,
    )

    # 4. Tile binning: cumsum -> map_gaussian_to_intersects -> sort -> get_tile_bin_edges
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()

    if num_intersects == 0:
        # No visible gaussians, return background
        out = background.unsqueeze(0).unsqueeze(0).expand(img_height, img_width, 3).contiguous()
        return out

    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        N, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds,
    )

    sorted_vals = torch.sort(isect_ids)
    isect_ids_sorted = sorted_vals[0]
    sort_indices = sorted_vals[1]
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sort_indices)

    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)

    # 5. Rasterize
    block = (BLOCK_X, BLOCK_Y, 1)
    img_size = (img_width, img_height, 1)

    out_img, final_Ts, final_idx = nd_rasterize_forward(
        tile_bounds, block, img_size,
        gaussian_ids_sorted, tile_bins,
        xys, conics, colors, opacities, background,
    )

    return out_img  # (H, W, 3)
