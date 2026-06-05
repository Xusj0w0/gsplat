#include <cmath>
#include <filesystem>
#include <iostream>
#include <tuple>

#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "Intersect.h"
#include "Ops.h"
#include "utils/utils.h"

namespace {

torch::Tensor cudaFloatContiguous(torch::Tensor t) {
    return t.to(torch::kCUDA, torch::kFloat32).contiguous();
}

torch::Tensor makePinholeK(const Snapshot &s) {
    TORCH_CHECK(s.tanfovx > 0.0f, "tanfovx must be positive");
    TORCH_CHECK(s.tanfovy > 0.0f, "tanfovy must be positive");

    const float fx = static_cast<float>(s.image_width) / (2.0f * s.tanfovx);
    const float fy = static_cast<float>(s.image_height) / (2.0f * s.tanfovy);
    const float cx = static_cast<float>(s.image_width) * 0.5f;
    const float cy = static_cast<float>(s.image_height) * 0.5f;

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return torch::tensor(
        {{{fx, 0.0f, cx}, {0.0f, fy, cy}, {0.0f, 0.0f, 1.0f}}}, options
    );
}

torch::Tensor asSingleCameraViewmat(torch::Tensor viewmatrix) {
    viewmatrix = cudaFloatContiguous(viewmatrix.transpose(-1, -2)
    ); // ViewMatrix in snapshot is transposed
    if (viewmatrix.dim() == 2) {
        TORCH_CHECK(
            viewmatrix.size(0) == 4 && viewmatrix.size(1) == 4,
            "viewmatrix must have shape [4, 4] or [C, 4, 4]"
        );
        return viewmatrix.unsqueeze(0);
    }

    TORCH_CHECK(
        viewmatrix.dim() == 3 && viewmatrix.size(-2) == 4 &&
            viewmatrix.size(-1) == 4,
        "viewmatrix must have shape [4, 4] or [C, 4, 4]"
    );
    return viewmatrix;
}

torch::Tensor asGaussianColors(torch::Tensor colors, int64_t n_gaussians) {
    colors = cudaFloatContiguous(colors);
    if (colors.dim() == 3 && colors.size(0) == 1) {
        colors = colors.squeeze(0);
    }
    if (colors.dim() == 2 && colors.size(0) == 3 &&
        colors.size(1) == n_gaussians) {
        colors = colors.transpose(0, 1).contiguous();
    }
    TORCH_CHECK(
        colors.dim() == 2 && colors.size(0) == n_gaussians,
        "colors_precomp must have shape [N, C], [1, N, C], or [C, N]"
    );
    TORCH_CHECK(
        colors.size(1) == 3 || colors.size(1) == 4,
        "debug runner expects RGB/RGBA colors"
    );
    return colors;
}

torch::Tensor asGaussianSHs(torch::Tensor shs, int64_t n_gaussians) {
    shs = cudaFloatContiguous(shs);
    TORCH_CHECK(
        shs.dim() == 3 && shs.size(0) == n_gaussians && shs.size(2) == 3,
        "sh_precomp must have shape [N, :, 3]"
    );
    TORCH_CHECK(
        shs.size(1) == 1 || shs.size(1) == 4 || shs.size(1) == 9 ||
            shs.size(1) == 16,
        "sh_precomp must have shape [N, 1, 3], [N, 4, 3], [N, 9, 3], or [N, "
        "16, 3]"
    );
    return shs;
}

int shDegree(torch::Tensor shs) {
    int n_coeffs = shs.size(1);
    if (n_coeffs == 1)
        return 0;
    if (n_coeffs == 4)
        return 1;
    if (n_coeffs == 9)
        return 2;
    if (n_coeffs == 16)
        return 3;
    TORCH_CHECK(false, "Unsupported number of SH coefficients: ", n_coeffs);
    return -1; // Unreachable
}

torch::Tensor
asGaussianScalars(torch::Tensor values, int64_t n_gaussians, const char *name) {
    values = cudaFloatContiguous(values);
    if (values.dim() == 2 && values.size(1) == 1) {
        values = values.squeeze(1);
    }
    if (values.dim() == 2 && values.size(0) == 1 &&
        values.size(1) == n_gaussians) {
        values = values.squeeze(0);
    }
    TORCH_CHECK(
        values.dim() == 1 && values.size(0) == n_gaussians,
        name,
        " must have shape [N] or [N, 1]"
    );
    return values.contiguous();
}

at::optional<torch::Tensor>
optionalCovars(const Snapshot &s, int64_t n_gaussians) {
    if (!s.cov3ds_precomp.defined() || s.cov3ds_precomp.numel() == 0) {
        return c10::nullopt;
    }

    torch::Tensor covars = cudaFloatContiguous(s.cov3ds_precomp);
    if (covars.dim() == 3 && covars.size(0) == 1) {
        covars = covars.squeeze(0);
    }
    TORCH_CHECK(
        covars.dim() == 2 && covars.size(0) == n_gaussians &&
            covars.size(1) == 6,
        "cov3ds_precomp must have shape [N, 6] when provided"
    );
    return covars.contiguous();
}

} // namespace

int main(int argc, char **argv) {
    torch::NoGradGuard no_grad;

    std::filesystem::path snapshot_path =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::path(
                       "/mnt/d/projects/cuda/gaussian-rasterization/snapshot/"
                       "snapshot.pt"
                   );
    std::filesystem::path output_path =
        argc > 2 ? std::filesystem::path(argv[2])
                 : snapshot_path.parent_path() / "gsplat_debug_render.png";

    if (!std::filesystem::exists(snapshot_path)) {
        std::cerr << "Snapshot not found: " << snapshot_path << std::endl;
        return 1;
    }

    try {
        std::cout << "Loading snapshot: " << snapshot_path << std::endl;
        torch::jit::Module module =
            torch::jit::load(snapshot_path.string(), torch::kCUDA);
        Snapshot s = loadSnapshotFromModule(module);

        TORCH_CHECK(torch::cuda::is_available(), "CUDA is not available");
        TORCH_CHECK(
            s.image_width > 0 && s.image_height > 0, "Invalid image size"
        );

        const int64_t n_gaussians = s.means3d.size(0);
        const int64_t tile_size = 16;
        const int64_t tile_width = (s.image_width + tile_size - 1) / tile_size;
        const int64_t tile_height =
            (s.image_height + tile_size - 1) / tile_size;
        const int64_t n_images = 1;

        torch::Tensor means = cudaFloatContiguous(s.means3d);
        torch::Tensor shs = asGaussianSHs(s.sh, n_gaussians);
        int sh_degree = shDegree(shs);
        torch::Tensor opacities =
            asGaussianScalars(s.opacities, n_gaussians, "opacities");
        torch::Tensor viewmats = asSingleCameraViewmat(s.viewmatrix);
        torch::Tensor Ks = makePinholeK(s);

        at::optional<torch::Tensor> covars = optionalCovars(s, n_gaussians);
        at::optional<torch::Tensor> quats = c10::nullopt;
        at::optional<torch::Tensor> scales = c10::nullopt;
        if (!covars.has_value()) {
            quats = cudaFloatContiguous(s.rotations);
            scales = cudaFloatContiguous(s.scales) * s.scale_modifier;
        }

        constexpr double eps2d = 0.3;
        constexpr double near_plane = 0.01;
        constexpr double far_plane = 1.0e10;
        constexpr double radius_clip = 0.0;
        constexpr bool calc_compensations = false;
        constexpr int64_t camera_model_pinhole = 0;

        std::cout << "Running projection for " << n_gaussians << " gaussians"
                  << std::endl;
        auto
            [radii,
             means2d,
             depths,
             conics,
             compensations,
             ray_planes,
             normals] =
                gsplat::projection_radegs_fused_fwd(
                    means,
                    covars,
                    quats,
                    scales,
                    opacities,
                    viewmats,
                    Ks,
                    s.image_width,
                    s.image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    calc_compensations,
                    camera_model_pinhole
                );

        std::cout << "Computing tile intersections" << std::endl;
        auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
            means2d,
            radii,
            depths,
            c10::nullopt,
            c10::nullopt,
            n_images,
            tile_size,
            tile_width,
            tile_height,
            true,
            false
        );
        torch::Tensor tile_offsets = gsplat::intersect_offset(
            isect_ids, n_images, tile_width, tile_height
        );

        torch::Tensor background = cudaFloatContiguous(s.bg);
        if (background.dim() == 1) {
            background = background.unsqueeze(0);
        }
        std::cout << "Decoding colors from SH coefficients" << std::endl;
        torch::Tensor campos = torch::linalg_inv(viewmats)
                                   .index({torch::indexing::Slice(0, 3), -1})
                                   .contiguous();
        torch::Tensor dirs = means - campos.unsqueeze(0);
        torch::Tensor mask = radii > 0.0;
        torch::Tensor colors =
            gsplat::spherical_harmonics_fwd(sh_degree, dirs, shs, mask);

        std::cout << "Rasterizing " << isect_ids.numel() << " intersections"
                  << std::endl;
        auto [render_colors, render_alphas, geometries, median_ids, last_ids] =
            gsplat::rasterize_to_pixels_radegs_fwd(
                means2d,
                conics,
                colors.unsqueeze(0),
                opacities.unsqueeze(0),
                ray_planes,
                normals,
                background,
                c10::nullopt,
                s.image_width,
                s.image_height,
                tile_size,
                tile_offsets,
                flatten_ids
            );

        c10::cuda::getCurrentCUDAStream().synchronize();

        torch::Tensor image = render_colors[0].clamp(0.0, 1.0);
        if (!saveTensorPNG(image, output_path.string())) {
            std::cerr << "Failed to save PNG: " << output_path << std::endl;
            return 1;
        }

        std::cout << "Saved debug render: " << output_path << std::endl;
        return 0;
    } catch (const c10::Error &e) {
        std::cerr << "Torch error: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 1;
}
