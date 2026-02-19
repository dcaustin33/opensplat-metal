"""Smoke test: verify the C extension loads and basic functions work."""
import torch
import opensplat_metal


def test_constants():
    assert opensplat_metal.BLOCK_X == 16
    assert opensplat_metal.BLOCK_Y == 16
    print("BLOCK_X =", opensplat_metal.BLOCK_X)
    print("BLOCK_Y =", opensplat_metal.BLOCK_Y)


def test_compute_sh_forward():
    device = "mps"
    N = 4
    degree = 0
    viewdirs = torch.randn(N, 3, device=device, dtype=torch.float32)
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    coeffs = torch.randn(N, 1, 3, device=device, dtype=torch.float32)
    colors = opensplat_metal.compute_sh_forward(N, degree, degree, viewdirs, coeffs)
    assert colors.shape == (N, 3), f"Expected (N, 3), got {colors.shape}"
    print("compute_sh_forward OK, colors shape:", colors.shape)


def test_project_gaussians_forward():
    device = "mps"
    N = 8
    means = torch.randn(N, 3, device=device, dtype=torch.float32) * 2.0
    means[:, 2] += 5.0  # push in front of camera
    scales = torch.ones(N, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.zeros(N, 4, device=device, dtype=torch.float32)
    quats[:, 0] = 1.0  # identity quaternion
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    projmat = torch.eye(4, device=device, dtype=torch.float32)

    tile_bounds = (32, 32, 1)
    cov3d, xys, depths, radii, conics, num_tiles_hit = opensplat_metal.project_gaussians_forward(
        N, means, scales, 1.0, quats, viewmat, projmat,
        500.0, 500.0, 256.0, 256.0, 512, 512, tile_bounds, 0.01,
    )
    print(f"project_gaussians_forward OK: xys {xys.shape}, radii {radii.shape}")
    print(f"  visible (radii > 0): {(radii > 0).sum().item()}/{N}")


if __name__ == "__main__":
    test_constants()
    test_compute_sh_forward()
    test_project_gaussians_forward()
    print("\nAll smoke tests passed!")
