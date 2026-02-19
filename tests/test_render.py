"""End-to-end render test using ~/Downloads/banana.ply gaussian splat model."""
import struct
import math
import numpy as np
import torch
import opensplat_metal
from pathlib import Path


def parse_ply(path):
    """Parse a gaussian splatting PLY file into component tensors."""
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Extract vertex count
        vertex_count = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        print(f"PLY: {vertex_count} vertices, {len(properties)} properties")
        prop_names = [p[0] for p in properties]
        print(f"Properties: {prop_names[:10]}... ({len(prop_names)} total)")

        # Read binary data
        dtype_map = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I"}
        struct_fmt = "<" + "".join(dtype_map.get(p[1], "f") for p in properties)
        record_size = struct.calcsize(struct_fmt)

        data = {}
        for name, _ in properties:
            data[name] = []

        for _ in range(vertex_count):
            record = struct.unpack(struct_fmt, f.read(record_size))
            for i, (name, _) in enumerate(properties):
                data[name].append(record[i])

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key], dtype=np.float32)

    return data, vertex_count


def extract_gaussians(data, vertex_count, device="mps"):
    """Extract gaussian parameters from parsed PLY data."""
    # Positions
    means = torch.tensor(
        np.stack([data["x"], data["y"], data["z"]], axis=-1),
        dtype=torch.float32, device=device,
    )

    # Scales (stored as log-scale, need exp)
    scales = torch.tensor(
        np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1),
        dtype=torch.float32, device=device,
    )
    scales = torch.exp(scales)

    # Rotations (quaternions)
    quats = torch.tensor(
        np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1),
        dtype=torch.float32, device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # Opacities (stored as logit, need sigmoid)
    opacities = torch.tensor(data["opacity"], dtype=torch.float32, device=device)
    opacities = torch.sigmoid(opacities)

    # SH coefficients
    # f_dc_0, f_dc_1, f_dc_2 are the DC components
    # f_rest_0 ... f_rest_N are higher-order SH
    sh_dc = torch.tensor(
        np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1),
        dtype=torch.float32, device=device,
    )  # (N, 3)

    # Find rest SH coefficients
    rest_keys = sorted(
        [k for k in data.keys() if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    num_rest = len(rest_keys)
    sh_degree = 0
    if num_rest >= 45:
        sh_degree = 3
    elif num_rest >= 24:
        sh_degree = 2
    elif num_rest >= 9:
        sh_degree = 1

    num_bases_map = {0: 1, 1: 4, 2: 9, 3: 16}
    num_bases = num_bases_map[sh_degree]
    num_rest_needed = (num_bases - 1) * 3

    # Build SH coefficients tensor: (N, num_bases, 3)
    sh_coeffs = torch.zeros(vertex_count, num_bases, 3, dtype=torch.float32, device=device)
    sh_coeffs[:, 0, :] = sh_dc

    if sh_degree > 0 and num_rest_needed > 0:
        rest_data = np.stack([data[k] for k in rest_keys[:num_rest_needed]], axis=-1)  # (N, num_rest_needed)
        rest_tensor = torch.tensor(rest_data, dtype=torch.float32, device=device)
        # Rest coefficients are stored as interleaved: for each basis, 3 channels
        rest_tensor = rest_tensor.reshape(vertex_count, num_bases - 1, 3)
        sh_coeffs[:, 1:num_bases, :] = rest_tensor

    print(f"Extracted: {vertex_count} gaussians, SH degree {sh_degree} ({num_bases} bases)")
    return means, quats, scales, sh_coeffs, opacities, sh_degree


def make_camera(img_w, img_h, fov_x_deg=60.0, cam_distance=4.0, elevation_deg=30.0, azimuth_deg=45.0, target=None, device="mps"):
    """Create a camera orbiting around a target point.

    Uses COLMAP/OpenCV convention: +X right, +Y down, +Z forward (into scene).
    The view matrix transforms world points into this camera space.
    """
    fov_x = math.radians(fov_x_deg)
    fx = 0.5 * img_w / math.tan(0.5 * fov_x)
    fy = fx  # square pixels
    cx = img_w / 2.0
    cy = img_h / 2.0

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    if target is None:
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Camera position: orbit around target
    # Elevation rotates up from the horizontal plane, azimuth rotates around y-axis
    # Using y-up convention (common in 3DGS scenes from COLMAP)
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    cam_x = cam_distance * math.cos(elev) * math.sin(azim)
    cam_y = -cam_distance * math.sin(elev)
    cam_z = cam_distance * math.cos(elev) * math.cos(azim)
    cam_pos = target + np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # Look-at: camera Z points from camera toward target
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    # World up = -Y (since Y is down in COLMAP)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Degenerate case: looking straight up/down
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # View matrix (world-to-camera): rows are camera axes in world space
    # Camera +X = right, +Y = down, +Z = forward
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ cam_pos

    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = torch.tensor(R, dtype=torch.float32, device=device)
    viewmat[:3, 3] = torch.tensor(t, dtype=torch.float32, device=device)

    return viewmat, K


def save_image(tensor, path):
    """Save (H, W, 3) float tensor as PNG."""
    img = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    img = (img * 255).astype(np.uint8)

    # Write PPM (simple, no dependency)
    h, w, c = img.shape
    ppm_path = str(path).rsplit(".", 1)[0] + ".ppm"
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())
    print(f"Saved PPM: {ppm_path}")

    # Try PNG via PIL if available
    try:
        from PIL import Image
        Image.fromarray(img).save(str(path))
        print(f"Saved PNG: {path}")
    except ImportError:
        print("PIL not available, only PPM saved")


def main():
    ply_path = Path.home() / "Downloads" / "banana.ply"
    if not ply_path.exists():
        print(f"ERROR: {ply_path} not found")
        return

    device = "mps"
    img_w, img_h = 800, 600

    print(f"Loading {ply_path}...")
    data, vertex_count = parse_ply(ply_path)
    means, quats, scales, sh_coeffs, opacities, sh_degree = extract_gaussians(data, vertex_count, device)

    # Compute scene center and extent for camera placement
    means_cpu = means.cpu().numpy()
    center = means_cpu.mean(axis=0)
    extent = np.percentile(np.linalg.norm(means_cpu - center, axis=1), 95)
    cam_distance = extent * 4.0
    print(f"Scene center: {center}, extent: {extent:.3f}, cam_distance: {cam_distance:.3f}")

    # Inspect per-axis spread to understand scene orientation
    pct_lo = np.percentile(means_cpu, 5, axis=0)
    pct_hi = np.percentile(means_cpu, 95, axis=0)
    print(f"Per-axis 5-95% range:  X [{pct_lo[0]:.3f}, {pct_hi[0]:.3f}]  "
          f"Y [{pct_lo[1]:.3f}, {pct_hi[1]:.3f}]  Z [{pct_lo[2]:.3f}, {pct_hi[2]:.3f}]")

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Render orbit views with DC-only SH (degree 0) to get clean base colors
    # (higher SH degrees produce view-dependent effects that look bad at novel views)
    for i, azimuth in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        viewmat, K = make_camera(
            img_w, img_h,
            fov_x_deg=60.0,
            cam_distance=cam_distance,
            elevation_deg=25.0,
            azimuth_deg=azimuth,
            target=center,
            device=device,
        )

        print(f"\nRendering view {i} (azimuth={azimuth})...")
        with torch.no_grad():
            out_img = opensplat_metal.render(
                means, quats, scales, sh_coeffs, opacities,
                viewmat, K, img_h, img_w,
                background=torch.ones(3, device=device),
                glob_scale=1.0,
                clip_thresh=0.01,
                sh_degree=0,  # DC-only for clean novel views
            )

        print(f"  Output shape: {out_img.shape}")
        print(f"  Value range: [{out_img.min().item():.4f}, {out_img.max().item():.4f}]")

        save_image(out_img, output_dir / f"banana_view_{i:02d}.png")

    print(f"\nDone! Output saved to {output_dir}/")


if __name__ == "__main__":
    main()
