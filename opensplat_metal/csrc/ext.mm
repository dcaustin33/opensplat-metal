#import "bindings.h"
#import "config.h"
#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
    m.attr("BLOCK_X") = BLOCK_X;
    m.attr("BLOCK_Y") = BLOCK_Y;

    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor,
          "Compute 2D covariance bounds",
          py::arg("num_pts"), py::arg("covs2d"));

    m.def("compute_sh_forward", &compute_sh_forward_tensor,
          "Compute SH forward (colors from SH coefficients)",
          py::arg("num_points"), py::arg("degree"), py::arg("degrees_to_use"),
          py::arg("viewdirs"), py::arg("coeffs"));

    m.def("project_gaussians_forward", &project_gaussians_forward_tensor,
          "Project 3D gaussians to 2D",
          py::arg("num_points"), py::arg("means3d"), py::arg("scales"),
          py::arg("glob_scale"), py::arg("quats"), py::arg("viewmat"),
          py::arg("projmat"), py::arg("fx"), py::arg("fy"),
          py::arg("cx"), py::arg("cy"),
          py::arg("img_height"), py::arg("img_width"),
          py::arg("tile_bounds"), py::arg("clip_thresh"));

    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor,
          "Map gaussians to tile intersections",
          py::arg("num_points"), py::arg("num_intersects"),
          py::arg("xys"), py::arg("depths"), py::arg("radii"),
          py::arg("num_tiles_hit"), py::arg("tile_bounds"));

    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor,
          "Get tile bin edges from sorted intersection IDs",
          py::arg("num_intersects"), py::arg("isect_ids_sorted"));

    m.def("rasterize_forward", &rasterize_forward_tensor,
          "Rasterize forward (3-channel)",
          py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
          py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
          py::arg("xys"), py::arg("conics"), py::arg("colors"),
          py::arg("opacities"), py::arg("background"));

    m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor,
          "Rasterize forward (N-channel)",
          py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
          py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
          py::arg("xys"), py::arg("conics"), py::arg("colors"),
          py::arg("opacities"), py::arg("background"));
}
