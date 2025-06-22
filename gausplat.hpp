
#include "spherical_harmonics_coefs.h"
#include "typedefs.h"
#include "utils.hpp"
#include <torch/torch.h>

namespace
{
torch::Tensor upperTriangular(const torch::Tensor &mat)
{
    using torch::indexing::Slice;
    int64_t s    = mat.size(1);
    auto    idx  = torch::triu_indices(s, s);
    auto    idx0 = idx[0];
    auto    idx1 = idx[1];
    return mat.index({Slice(), idx0, idx1});
}
} // namespace

namespace gsplat
{
using torch::indexing::Slice;
using torch::indexing::TensorIndex;

torch::Tensor GLOBAL_IDX_MAP;

// Projects 3d gaussians from world coordinates to camera coordinates.
// Also returns the image projection of the center of 3d gaussian
torch::Tensor project(torch::Tensor &points_world, const Camera &cam, torch::Tensor &proj_us)
{

    torch::Tensor points_cam = torch::matmul(points_world, cam.Rcw.transpose(0, 1)) + cam.tcw;

    torch::Tensor x = points_cam.index({torch::indexing::Slice(), 0});
    torch::Tensor y = points_cam.index({torch::indexing::Slice(), 1});
    torch::Tensor z = points_cam.index({torch::indexing::Slice(), 2});

    torch::Tensor u_x = x.mul(cam.fx).div(z).add(cam.cx);
    torch::Tensor u_y = y.mul(cam.fy).div(z).add(cam.cy);

    proj_us = torch::stack({u_x, u_y}, /*dim=*/1);

    return points_cam;
}

torch::Tensor computeCov3d(const torch::Tensor &scale, const torch::Tensor &rot)
{
    int64_t batch  = scale.size(0);
    auto    device = scale.device();
    auto    dtype  = scale.scalar_type();

    // Build diagonal scale matrix S: shape [batch, 3, 3]
    auto S = torch::zeros({batch, 3, 3}, torch::TensorOptions().device(device).dtype(dtype));
    S.index_put_({Slice(), 0, 0}, scale.index({Slice(), 0}));
    S.index_put_({Slice(), 1, 1}, scale.index({Slice(), 1}));
    S.index_put_({Slice(), 2, 2}, scale.index({Slice(), 2}));

    auto w = rot.index({Slice(), 0});
    auto x = rot.index({Slice(), 1});
    auto y = rot.index({Slice(), 2});
    auto z = rot.index({Slice(), 3});

    // Compute rotation matrix entries
    auto r00 = 1 - 2 * (y * y + z * z);
    auto r01 = 2 * (x * y - z * w);
    auto r02 = 2 * (x * z + y * w);
    auto r10 = 2 * (x * y + z * w);
    auto r11 = 1 - 2 * (x * x + z * z);
    auto r12 = 2 * (y * z - x * w);
    auto r20 = 2 * (x * z - y * w);
    auto r21 = 2 * (y * z + x * w);
    auto r22 = 1 - 2 * (x * x + y * y);

    // Stack rows into R: shape [batch, 3, 3]
    auto row0 = torch::stack({r00, r01, r02}, /*dim=*/1);
    auto row1 = torch::stack({r10, r11, r12}, /*dim=*/1);
    auto row2 = torch::stack({r20, r21, r22}, /*dim=*/1);
    auto R    = torch::stack({row0, row1, row2}, /*dim=*/1);

    // M = R @ S, Sigma = M @ M^T
    auto M     = torch::bmm(R, S);
    auto Sigma = torch::bmm(M, M.transpose(1, 2));

    // return upperTriangular(Sigma);
    return Sigma;
}

torch::Tensor projectCov3dTo2d(const torch::Tensor &points_cam, const Camera &cam, const torch::Tensor &cov3d)
{
    auto x = points_cam.index({Slice(), 0}); // [batch]
    auto y = points_cam.index({Slice(), 1}); // [batch]
    auto z = points_cam.index({Slice(), 2}); // [batch]

    // Compute field‐of‐view limits
    float tan_fovx = 2.0f * std::atan(cam.width / (2.0f * cam.fx));
    float tan_fovy = 2.0f * std::atan(cam.height / (2.0f * cam.fy));
    float limx     = 1.3f * tan_fovx;
    float limy     = 1.3f * tan_fovy;

    auto inv_z_x = x.div(z);
    inv_z_x      = torch::clamp(inv_z_x, -limx, limx);
    x            = inv_z_x.mul(z);
    auto inv_z_y = y.div(z);
    inv_z_y      = torch::clamp(inv_z_y, -limy, limy);
    y            = inv_z_y.mul(z);

    int64_t batch  = points_cam.size(0);
    auto    device = points_cam.device();
    auto    dtype  = points_cam.scalar_type();

    // Build Jacobian J: [batch, 3, 3], initialized to zeros
    auto J = torch::zeros({batch, 3, 3}, torch::TensorOptions().device(device).dtype(dtype));
    // J[:, 0, 0] = fx / z
    J.index_put_({Slice(), 0, 0}, cam.fx / z);
    // J[:, 0, 2] = -(fx * x) / (z*z)
    J.index_put_({Slice(), 0, 2}, -(cam.fx * x).div(z * z));
    // J[:, 1, 1] = fy / z
    J.index_put_({Slice(), 1, 1}, cam.fy / z);
    // J[:, 1, 2] = -(fy * y) / (z*z)
    J.index_put_({Slice(), 1, 2}, -(cam.fy * y).div(z * z));

    auto T     = torch::matmul(J, cam.Rcw); // [batch, 3, 3]
    auto Sigma = cov3d;
    // Sigma' = T @ Sigma @ Tᵀ
    auto TSigma      = torch::bmm(T, Sigma);
    auto Sigma_prime = torch::bmm(TSigma, T.transpose(1, 2));

    // inflate x & y variance by 0.3
    {
        auto diag00 = Sigma_prime.index({Slice(), 0, 0});
        Sigma_prime.index_put_({Slice(), 0, 0}, diag00 + 0.3f);
        auto diag11 = Sigma_prime.index({Slice(), 1, 1});
        Sigma_prime.index_put_({Slice(), 1, 1}, diag11 + 0.3f);
    }

    // Extract the top‐left 2×2 block:
    auto Sigma2d = Sigma_prime.index({Slice(), Slice(0, 2), Slice(0, 2)});
    return upperTriangular(Sigma2d);
}

torch::Tensor shToColor(torch::Tensor       &sh, // [batch, sh_dim]
                        torch::Tensor       &pw, // [batch, 3]
                        const torch::Tensor &twc // [batch, 3]
)
{
    int64_t sh_dim = sh.size(1);

    auto color = SH_C0_0 * sh.index({Slice(), Slice(0, 3)}) + 0.5;
    if (sh_dim <= 3)
    {
        return color;
    }

    auto ray_dir = pw - twc; // [batch, 3]
    ray_dir      = ray_dir / ray_dir.norm(2, /*dim=*/1, /*keepdim=*/true);

    auto x = ray_dir.index({Slice(), Slice(0, 1)}); // [batch,1]
    auto y = ray_dir.index({Slice(), Slice(1, 2)}); // [batch,1]
    auto z = ray_dir.index({Slice(), Slice(2, 3)}); // [batch,1]

    color = color + SH_C1_0 * (y * sh.index({Slice(), Slice(3, 6)})) +
            SH_C1_1 * (z * sh.index({Slice(), Slice(6, 9)})) + SH_C1_2 * (x * sh.index({Slice(), Slice(9, 12)}));
    if (sh_dim <= 12)
    {
        return color;
    }

    // compute second‐order terms
    auto xx = x * x; // [batch,1]
    auto yy = y * y;
    auto zz = z * z;
    auto xy = x * y;
    auto yz = y * z;
    auto xz = x * z;

    color = color + SH_C2_0 * (xy * sh.index({Slice(), Slice(12, 15)})) +
            SH_C2_1 * (yz * sh.index({Slice(), Slice(15, 18)})) +
            SH_C2_2 * ((2.0 * zz - xx - yy) * sh.index({Slice(), Slice(18, 21)})) +
            SH_C2_3 * (xz * sh.index({Slice(), Slice(21, 24)})) +
            SH_C2_4 * ((xx - yy) * sh.index({Slice(), Slice(24, 27)}));
    if (sh_dim <= 27)
    {
        return color;
    }

    color = color + SH_C3_0 * (y * (3.0 * xx - yy) * sh.index({Slice(), Slice(27, 30)})) +
            SH_C3_1 * (xy * z * sh.index({Slice(), Slice(30, 33)})) +
            SH_C3_2 * (y * ((4.0 * zz - xx - yy) * sh.index({Slice(), Slice(33, 36)}))) +
            SH_C3_3 * (z * ((2.0 * zz - 3.0 * xx - 3.0 * yy) * sh.index({Slice(), Slice(36, 39)}))) +
            SH_C3_4 * (x * ((4.0 * zz - xx - yy) * sh.index({Slice(), Slice(39, 42)}))) +
            SH_C3_5 * (z * ((xx - yy) * sh.index({Slice(), Slice(42, 45)}))) +
            SH_C3_6 * (x * ((xx - 3.0 * yy) * sh.index({Slice(), Slice(45, 48)})));

    return color; // [batch,3]
}

// Computes the inverse of 2×2 covariances stored as [cov_xx, cov_xy, cov_yy] per row.
// Returns:
//  - cinv: [batch, 3] holding [cov_yy * det_inv, -cov_xy * det_inv, cov_xx * det_inv]
//  - areas: [batch, 2] holding [3 * sqrt(cov_xx), 3 * sqrt(cov_yy)] cast to int32
torch::Tensor inverseCov2d(const torch::Tensor &cov2d, torch::Tensor &areas_3_sigma)
{
    // cov2d: [batch, 3]
    auto cov_xx = cov2d.index({Slice(), 0}); // [batch]
    auto cov_xy = cov2d.index({Slice(), 1}); // [batch]
    auto cov_yy = cov2d.index({Slice(), 2}); // [batch]

    auto det_term = cov_xx * cov_yy - cov_xy * cov_xy + 1e-6;
    auto det_inv  = det_term.reciprocal(); // [batch]

    auto c00       = cov_yy.mul(det_inv);
    auto c01       = cov_xy.mul(det_inv).neg();
    auto c11       = cov_xx.mul(det_inv);
    auto cov2d_inv = torch::stack({c00, c01, c11}, /*dim=*/1); // [batch, 3]

    auto stacked  = torch::stack({cov_xx, cov_yy}, /*dim=*/1); // [batch, 2]
    auto sqrted   = torch::sqrt(stacked);
    areas_3_sigma = sqrted.mul(3.0).to(torch::kInt32); // [batch, 2]

    return cov2d_inv;
}

torch::Tensor splat(const Camera  &cam,
                    torch::Tensor &gaussian_2d_centers,
                    torch::Tensor &cov2d_inv,
                    torch::Tensor &alphas,
                    torch::Tensor &depths,
                    torch::Tensor &colors,
                    torch::Tensor &areas_3_sigma)
{
    // Create an empty RGB image on the same device and dtype as the inputs:
    auto          device = gaussian_2d_centers.device();
    auto          dtype  = gaussian_2d_centers.dtype();
    torch::Tensor image  = torch::zeros({cam.height, cam.width, 3}, torch::TensorOptions().device(device).dtype(dtype));
    torch::Tensor image_T = torch::ones({cam.height, cam.width}, torch::TensorOptions().device(device).dtype(dtype));

    // Sort indices by depth (ascending)
    torch::Tensor depth_sorted_idxs = depths.argsort();
    int64_t       num_gaussians     = depths.size(0);

    for (int64_t k = 0; k < num_gaussians; ++k)
    {
        const int64_t idx = depth_sorted_idxs[k].item<int64_t>();
        // Fetch scalar depth value
        const float dpt = depths.index({idx}).item<float>();
        if (dpt < 0.2f || dpt > 100.0f)
        {
            continue; // skip if out of valid depth range
        }
        // Fetch the 2D Gaussian center (gx, gy)
        auto gaussian_2d_center = gaussian_2d_centers.index({idx});
        auto gx_tensor          = gaussian_2d_center.index({0});
        auto gy_tensor          = gaussian_2d_center.index({1});
        // But we also need float values to compute integer bounds:
        const float gx = gx_tensor.item<float>();
        const float gy = gy_tensor.item<float>();

        // Quickly cull if normalized center is out of ±1.3 range
        if (std::abs(gx / cam.width) > 1.3f || std::abs(gy / cam.height) > 1.3f)
        {
            continue;
        }

        // Fetch the integer 3-sigma “radius” in pixels: [r_x, r_y]
        int r_x = areas_3_sigma.index({idx, 0}).item<int>();
        int r_y = areas_3_sigma.index({idx, 1}).item<int>();

        // Compute bounding box [x0,x1) × [y0,y1) of this gaussian in image space, clamped
        int x0 = static_cast<int>(std::clamp(gx - r_x, 0.0f, float(cam.width)));
        int x1 = static_cast<int>(std::clamp(gx + r_x, 0.0f, float(cam.width)));
        int y0 = static_cast<int>(std::clamp(gy - r_y, 0.0f, float(cam.height)));
        int y1 = static_cast<int>(std::clamp(gy + r_y, 0.0f, float(cam.height)));

        if (x1 <= x0 || y1 <= y0)
        {
            continue; // invalid patch
        }

        // Slice out the sub-region that this gaussian will affect from GLOBAL_IDX_MAP
        auto sub_region = GLOBAL_IDX_MAP.index({
            Slice(),       // both channels
            Slice(y0, y1), // rows
            Slice(x0, x1)  // cols
        });
        auto sub_x      = sub_region.index({0}); // [patch_h, patch_w], X‐coordinates
        auto sub_y      = sub_region.index({1}); // [patch_h, patch_w], Y‐coordinates

        // Pull out inverse-covariance scalars for this pixel: [c00, c01, c11]
        auto cinv00 = cov2d_inv.index({idx, 0}); // 0-d Tensor
        auto cinv01 = cov2d_inv.index({idx, 1}); // 0-d Tensor
        auto cinv11 = cov2d_inv.index({idx, 2}); // 0-d Tensor

        // Compute Mahalanobis distance using Tensors:
        // d0 = (x - gx), d1 = (y - gy)
        auto d0   = sub_x - gx_tensor; // [patch_h, patch_w], still in graph
        auto d1   = sub_y - gy_tensor; // [patch_h, patch_w]
        auto maha = cinv00 * (d0 * d0) + cinv11 * (d1 * d1) + 2.0f * cinv01 * (d0 * d1); // [patch_h, patch_w]

        auto alpha       = alphas.index({idx});
        auto patch_alpha = torch::exp(-0.5f * maha).mul(alpha).clamp_max(0.99f); // [dy,dx]
        auto color       = colors.index({idx}).view({1, 1, 3});

        auto roi            = image.slice(0, y0, y1).slice(1, x0, x1);
        auto roi_T          = image_T.slice(0, y0, y1).slice(1, x0, x1);
        auto T              = roi_T.clone();
        auto weighted_alpha = patch_alpha * T;
        auto patch          = weighted_alpha.unsqueeze(-1).mul(color);
        roi.add_(patch);
        roi_T.mul_(1.0f - patch_alpha);
    }

    return image;
}

torch::Tensor splatTiled(const Camera  &cam,
                         torch::Tensor &gaussian_2d_centers,
                         torch::Tensor &cov2d_inv,
                         torch::Tensor &alphas,
                         torch::Tensor &depths,
                         torch::Tensor &colors,
                         torch::Tensor &areas_3_sigma)
{
    // Create an empty RGB image on the same device and dtype as the inputs:
    auto          device = gaussian_2d_centers.device();
    auto          dtype  = gaussian_2d_centers.dtype();
    torch::Tensor image  = torch::zeros({cam.height, cam.width, 3}, torch::TensorOptions().device(device).dtype(dtype));
    torch::Tensor image_T = torch::ones({cam.height, cam.width}, torch::TensorOptions().device(device).dtype(dtype));

    // Sort indices by depth (ascending)
    torch::Tensor depth_sorted_idxs = depths.argsort();
    int64_t       num_gaussians     = depths.size(0);

    constexpr int TW{64}; // tile width
    constexpr int TH{64}; // tile height
    const int     n_tiles_x = (cam.width + TW - 1) / TW;
    const int     n_tiles_y = (cam.height + TH - 1) / TH;

    // Precompute which gaussians belong to which tiles
    std::vector<std::vector<int32_t>> gaussians_in_tiles(n_tiles_x * n_tiles_y);
    for (int64_t k = 0; k < num_gaussians; ++k)
    {
        // Should still iterate the gaussians based on depth priority, in each tile
        const int64_t i = depth_sorted_idxs[k].item<int64_t>();

        const auto  gaussian_2d_center = gaussian_2d_centers.index({i});
        const float gx                 = gaussian_2d_center.index({0}).item<float>();
        const float gy                 = gaussian_2d_center.index({1}).item<float>();
        const int   r_x                = areas_3_sigma.index({i, 0}).item<int>();
        const int   r_y                = areas_3_sigma.index({i, 1}).item<int>();

        int x0  = std::max(0, int(std::floor(gx - r_x)));
        int x1  = std::min(int(cam.width), int(std::ceil(gx + r_x)));
        int y0  = std::max(0, int(std::floor(gy - r_y)));
        int y1  = std::min(int(cam.height), int(std::ceil(gy + r_y)));
        int tx0 = std::clamp(x0 / TW, 0, n_tiles_x - 1);
        int tx1 = std::clamp((x1 - 1) / TW, 0, n_tiles_x - 1);
        int ty0 = std::clamp(y0 / TH, 0, n_tiles_y - 1);
        int ty1 = std::clamp((y1 - 1) / TH, 0, n_tiles_y - 1);

        // Assing the gaussian to the tiles it affects
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx)
                gaussians_in_tiles[ty * n_tiles_x + tx].push_back(static_cast<int32_t>(i));
    }

    //--- process each tile
    for (int ty = 0; ty < n_tiles_y; ++ty)
    {
        const int y0_t = ty * TH;
        const int y1_t = std::min(int(cam.height), y0_t + TH);
        for (int tx = 0; tx < n_tiles_y; ++tx)
        {
            const int x0_t = tx * TW;
            const int x1_t = std::min(int(cam.width), x0_t + TW);

            // ROI for this tile. Tiled gaussians will affect this region of the image only
            auto roi   = image.slice(0, y0_t, y1_t).slice(1, x0_t, x1_t);
            auto roi_T = image_T.slice(0, y0_t, y1_t).slice(1, x0_t, x1_t);

            // Iterate over the gaussians in this tile
            for (int &gaus_idx : gaussians_in_tiles[ty * n_tiles_x + tx])
            {
                // Early bound checks
                const float depth{depths.index({gaus_idx}).item<float>()};
                if (depth < 0.2f || depth > 100.0f)
                {
                    continue;
                }
                auto        gaussian_2d_center = gaussian_2d_centers.index({gaus_idx});
                const float gx                 = gaussian_2d_center[0].item<float>();
                const float gy                 = gaussian_2d_center[1].item<float>();
                if (std::abs(gx / cam.width) > 1.3f || std::abs(gy / cam.height) > 1.3f)
                    continue;

                const int rx = areas_3_sigma[gaus_idx][0].item<int>();
                const int ry = areas_3_sigma[gaus_idx][1].item<int>();
                // clamp patch to this tile
                const int x0 = std::max(x0_t, int(std::floor(gx - rx)));
                const int x1 = std::min(x1_t, int(std::ceil(gx + rx)));
                const int y0 = std::max(y0_t, int(std::floor(gy - ry)));
                const int y1 = std::min(y1_t, int(std::ceil(gy + ry)));

                if (x1 <= x0 || y1 <= y0)
                    continue;

                auto sub_region = GLOBAL_IDX_MAP.index({Slice(), Slice(y0, y1), Slice(x0, x1)});
                auto sub_x      = sub_region.index({0}); // [patch_h, patch_w], X‐coordinates
                auto sub_y      = sub_region.index({1}); // [patch_h, patch_w], Y‐coordinates

                // Pull out inverse-covariance scalars for this pixel: [c00, c01, c11]
                auto cinv00 = cov2d_inv.index({gaus_idx, 0}); // 0-d Tensor
                auto cinv01 = cov2d_inv.index({gaus_idx, 1}); // 0-d Tensor
                auto cinv11 = cov2d_inv.index({gaus_idx, 2}); // 0-d Tensor

                // Compute Mahalanobis distance using Tensors:
                // d0 = (x - gx), d1 = (y - gy)
                auto d0   = sub_x - gaussian_2d_center[0];                                       // [patch_h, patch_w]
                auto d1   = sub_y - gaussian_2d_center[1];                                       // [patch_h, patch_w]
                auto maha = cinv00 * (d0 * d0) + cinv11 * (d1 * d1) + 2.0f * cinv01 * (d0 * d1); // [patch_h, patch_w]

                auto alpha       = alphas.index({gaus_idx});
                auto patch_alpha = torch::exp(-0.5f * maha).mul(alpha).clamp_max(0.99f); // [dy,dx]
                auto color       = colors.index({gaus_idx}).view({1, 1, 3});

                auto T              = roi_T.slice(0, y0 - y0_t, y1 - y0_t).slice(1, x0 - x0_t, x1 - x0_t).clone();
                auto weighted_alpha = patch_alpha * T;
                auto patch          = weighted_alpha.unsqueeze(-1).mul(color);
                // Adjust for the offset our gaussian has in the tile
                roi.slice(0, y0 - y0_t, y1 - y0_t).slice(1, x0 - x0_t, x1 - x0_t).add_(patch);
                roi_T.slice(0, y0 - y0_t, y1 - y0_t).slice(1, x0 - x0_t, x1 - x0_t).mul_(1.0f - patch_alpha);
            }
        }
    }

    return image;
}

torch::Tensor splatTiledBatched(const Camera  &cam,
                                torch::Tensor &gaus_2d_centers,
                                torch::Tensor &cov2d_inv,
                                torch::Tensor &alphas,
                                torch::Tensor &depths,
                                torch::Tensor &colors,
                                torch::Tensor &areas_3_sigma)
{
    auto device = gaus_2d_centers.device();
    auto dtype  = gaus_2d_centers.dtype();

    // Initialize output tensors
    auto t_ops       = torch::TensorOptions().device(device).dtype(dtype);
    auto t_ops_int32 = torch::TensorOptions().device(device).dtype(torch::kInt32);

    torch::Tensor image   = torch::zeros({cam.height, cam.width, 3}, t_ops);
    torch::Tensor image_T = torch::ones({cam.height, cam.width}, t_ops);

    // Sort Gaussians by depth
    torch::Tensor depth_sorted_idxs = depths.argsort();

    constexpr int TW{64}; // Tile width
    constexpr int TH{64}; // Tile height
    const int     n_tiles_x = (cam.width + TW - 1) / TW;
    const int     n_tiles_y = (cam.height + TH - 1) / TH;

    // Precompute Gaussian tile assignments
    at::GradMode::set_enabled(false);

    const auto gaus_2d_centers_depth_sorted = gaus_2d_centers.index({depth_sorted_idxs}); // [N, 2]
    const auto areas_depth_sorted           = areas_3_sigma.index({depth_sorted_idxs});   // [N, 2]

    // Determine 2d gaussian boundaries, clamped to image bounds
    const auto gaus_x_min = torch::max(
        torch::floor(gaus_2d_centers_depth_sorted.index({Slice(), 0}) - areas_depth_sorted.index({Slice(), 0}))
            .to(torch::kInt32),
        torch::tensor({0}, t_ops_int32));
    const auto gaus_x_max = torch::min(
        torch::ceil(gaus_2d_centers_depth_sorted.index({Slice(), 0}) + areas_depth_sorted.index({Slice(), 0}))
            .to(torch::kInt32),
        torch::tensor({cam.width}, t_ops_int32));
    const auto gaus_y_min = torch::max(
        torch::floor(gaus_2d_centers_depth_sorted.index({Slice(), 1}) - areas_depth_sorted.index({Slice(), 1}))
            .to(torch::kInt32),
        torch::tensor({0}, t_ops_int32));
    const auto gaus_y_max = torch::min(
        torch::ceil(gaus_2d_centers_depth_sorted.index({Slice(), 1}) + areas_depth_sorted.index({Slice(), 1}))
            .to(torch::kInt32),
        torch::tensor({cam.height}, t_ops_int32));

    // Determine tile indices for each Gaussian, clamped w.r.t number of tiles
    auto gaus_tile_x_idx_min = torch::clamp(gaus_x_min.floor_divide(TW), 0, n_tiles_x - 1);
    auto gaus_tile_x_idx_max = torch::clamp((gaus_x_max - 1).floor_divide(TW), 0, n_tiles_x - 1);
    auto gaus_tile_y_idx_min = torch::clamp(gaus_y_min.floor_divide(TH), 0, n_tiles_y - 1);
    auto gaus_tile_y_idx_max = torch::clamp((gaus_y_max - 1).floor_divide(TH), 0, n_tiles_y - 1);

    at::GradMode::set_enabled(true);

    // Process each tile
    for (int tile_y_idx = 0; tile_y_idx < n_tiles_y; ++tile_y_idx)
    {
        const int tile_y_min = tile_y_idx * TH;
        const int tile_y_max = std::min(int(cam.height), tile_y_min + TH);
        const int H{tile_y_max - tile_y_min};
        for (int tile_x_idx = 0; tile_x_idx < n_tiles_x; ++tile_x_idx)
        {
            const int tile_x_min = tile_x_idx * TW;
            const int tile_x_max = std::min(int(cam.width), tile_x_min + TW);
            const int W{tile_x_max - tile_x_min};

            // Find Gaussians covering this tile
            auto gaus_area_tile_overlap_mask =
                (gaus_tile_x_idx_min <= tile_x_idx) & (gaus_tile_x_idx_max >= tile_x_idx) &
                (gaus_tile_y_idx_min <= tile_y_idx) & (gaus_tile_y_idx_max >= tile_y_idx);
            auto      gaus_ids_in_this_tile = depth_sorted_idxs.masked_select(gaus_area_tile_overlap_mask);
            const int num_gaus_in_this_tile{gaus_ids_in_this_tile.size(0)};
            if (num_gaus_in_this_tile == 0)
                continue;

            // Tile coordinates
            at::GradMode::set_enabled(false);
            const auto sub_region =
                GLOBAL_IDX_MAP.index({Slice(), Slice(tile_y_min, tile_y_max), Slice(tile_x_min, tile_x_max)});
            const auto tile_x = sub_region.index({0});
            const auto tile_y = sub_region.index({1});
            at::GradMode::set_enabled(true);

            // Select gaussians covering this tile
            auto gaus_2d_centers_tile = gaus_2d_centers.index({gaus_ids_in_this_tile}).to(torch::kHalf); // [G, 2]
            auto cov2d_inv_tile       = cov2d_inv.index({gaus_ids_in_this_tile}).to(torch::kHalf);       // [G, 3]
            auto alphas_tile          = alphas.index({gaus_ids_in_this_tile}).to(torch::kHalf);          // [G]
            auto colors_tile          = colors.index({gaus_ids_in_this_tile}).to(torch::kHalf);          // [G, 3]
            auto areas_tile           = areas_3_sigma.index({gaus_ids_in_this_tile}).to(torch::kHalf);   // [G, 2]

            // Compute bounding boxes of gaussians within the tile
            at::GradMode::set_enabled(false);
            const auto gaus_x0_coord =
                torch::max(torch::floor(gaus_2d_centers_tile.index({Slice(), 0}) - areas_tile.index({Slice(), 0})),
                           torch::tensor({static_cast<float>(tile_x_min)}, t_ops));
            const auto gaus_x1_coord =
                torch::min(torch::ceil(gaus_2d_centers_tile.index({Slice(), 0}) + areas_tile.index({Slice(), 0})),
                           torch::tensor({static_cast<float>(tile_x_max)}, t_ops));
            const auto gaus_y0_coord =
                torch::max(torch::floor(gaus_2d_centers_tile.index({Slice(), 1}) - areas_tile.index({Slice(), 1})),
                           torch::tensor({static_cast<float>(tile_y_min)}, t_ops));
            const auto gaus_y1_coord =
                torch::min(torch::ceil(gaus_2d_centers_tile.index({Slice(), 1}) + areas_tile.index({Slice(), 1})),
                           torch::tensor({static_cast<float>(tile_y_max)}, t_ops));

            // Create bounding box mask
            const auto tile_x_coords       = tile_x.view({1, H, W});         // [1, H, W]
            const auto tile_y_coords       = tile_y.view({1, H, W});         // [1, H, W]
            const auto gaus_x0_coord_broad = gaus_x0_coord.view({-1, 1, 1}); // [G, 1, 1]
            const auto gaus_x1_coord_broad = gaus_x1_coord.view({-1, 1, 1});
            const auto gaus_y0_coord_broad = gaus_y0_coord.view({-1, 1, 1});
            const auto gaus_y1_coord_broad = gaus_y1_coord.view({-1, 1, 1});
            const auto mask_bb = ((tile_x_coords >= gaus_x0_coord_broad) & (tile_x_coords < gaus_x1_coord_broad) &
                                  (tile_y_coords >= gaus_y0_coord_broad) & (tile_y_coords < gaus_y1_coord_broad))
                                     .to(torch::kHalf);
            at::GradMode::set_enabled(true);

            // Compute differences for Mahalanobis distance
            auto       gx = gaus_2d_centers_tile.select(1, 0).unsqueeze(-1).unsqueeze(-1); // [G,1,1]
            auto       gy = gaus_2d_centers_tile.select(1, 1).unsqueeze(-1).unsqueeze(-1); // [G,1,1]
            const auto d0 = (tile_x_coords - gx).to(torch::kHalf);                         // [G, H, W]
            const auto d1 = (tile_y_coords - gy).to(torch::kHalf);                         // [G, H, W]

            // Compute Mahalanobis distance
            auto cinv00 = cov2d_inv_tile.index({Slice(), 0}).view({-1, 1, 1}).to(torch::kHalf); // [G, 1, 1]
            auto cinv01 = cov2d_inv_tile.index({Slice(), 1}).view({-1, 1, 1}).to(torch::kHalf);
            auto cinv11 = cov2d_inv_tile.index({Slice(), 2}).view({-1, 1, 1}).to(torch::kHalf);
            auto maha   = cinv00 * d0 * d0 + 2 * cinv01 * d0 * d1 + cinv11 * d1 * d1; // [G, H, W]

            // Compute alpha with masking
            auto alphas_tile_broad = alphas_tile.view({-1, 1, 1});                                   // [G, 1, 1]
            auto patch_alpha       = (torch::exp(-0.5 * maha) * alphas_tile_broad).to(torch::kHalf); // [G, H, W]
            patch_alpha            = patch_alpha * mask_bb; // Zero outside bounding box
            patch_alpha            = torch::clamp(patch_alpha, 0.0f, 0.99f);

            // // Accumulate contributions
            auto T_after  = torch::cumprod(1.0f - patch_alpha, 0); // [G, H, W]
            auto T_before = torch::cat(
                {torch::ones_like(patch_alpha.slice(0, 0, 1)), T_after.slice(0, 0, num_gaus_in_this_tile - 1)},
                0);                                                                                      // [G, H, W]
            auto contrib     = (patch_alpha * T_before).unsqueeze(-1) * colors_tile.view({-1, 1, 1, 3}); // [G, H, W, 3]
            auto final_color = contrib.sum(0);                                                           // [H, W, 3]
            auto final_T     = T_after.select(0, num_gaus_in_this_tile - 1);                             // [H, W]

            // Update image
            image.index_put_({Slice(tile_y_min, tile_y_max), Slice(tile_x_min, tile_x_max)}, final_color);
            image_T.index_put_({Slice(tile_y_min, tile_y_max), Slice(tile_x_min, tile_x_max)}, final_T);
        }
    }

    return image;
}

torch::Tensor forwardWithCulling(LearnableParams &params, const Camera &cam)
{
    auto alphas = getAlphas(params.alphas_raw);
    auto scales = getScales(params.scales_raw);
    auto rots   = getRots(params.rots_raw);
    auto shs    = getShs(params.low_shs, params.high_shs);

    // 1) Project 3D gaussian centers to camera coordinates and image coordinates
    torch::Tensor gaus_centers_img_frame;
    auto          gaus_centers_cam_frame = project(params.pws, cam, gaus_centers_img_frame);

    // 2) Build frustum mask and filter out gaus_ outside the frustum
    auto z = gaus_centers_cam_frame.index({Slice(), 2}); // [N]
    // Calculate normalized device coordinates of the gaussian centers in the image frame
    auto u_ndc    = gaus_centers_img_frame.index({Slice(), 0}) / cam.width * 2.F - 1.F;  // map to [-1, 1]
    auto v_ndc    = gaus_centers_img_frame.index({Slice(), 1}) / cam.height * 2.F - 1.F; // map to [-1, 1]
    auto radii_3d = std::get<0>(scales.max(1, false)) * 3.0f;   // 3 sigma based on gaussian covariance
    auto r_x_ndc  = radii_3d * (cam.fx / z) / cam.width * 2.0f; // radius in normalized image coordinates
    auto r_y_ndc  = radii_3d * (cam.fy / z) / cam.height * 2.0f;
    // Check if within the camera frustum (6 sides)
    auto frustum_cull_mask = (z > 0.2f) & (z < 100.f) & (u_ndc + r_x_ndc > -1) & (u_ndc - r_x_ndc < 1) &
                             (v_ndc + r_y_ndc > -1) & (v_ndc - r_y_ndc < 1);
    auto culled_gaus_ids = frustum_cull_mask.nonzero().squeeze(1); // [M]
    std::cout << "Survived Gaussians after Frustum Culling: " << culled_gaus_ids.size(0) << " / " << params.pws.size(0)
              << std::endl;

    // Pick the parameters of the surviving Gaussians (causes copy)
    auto pws_culled                    = params.pws.index({culled_gaus_ids});             // [M,3]
    auto alphas_culled                 = alphas.index({culled_gaus_ids});                 // [M]
    auto scales_culled                 = scales.index({culled_gaus_ids});                 // [M,3]
    auto rots_culled                   = rots.index({culled_gaus_ids});                   // [M,3,3]
    auto gaus_centers_img_frame_culled = gaus_centers_img_frame.index({culled_gaus_ids}); // [M,2]
    auto shs_culled                    = shs.index({culled_gaus_ids});                    // [M, sh_dim]

    // 3) Compute 3D covariance matrices for each gaussian
    auto cov3d_culled = computeCov3d(scales_culled, rots_culled);
    // 4) Project 3d gaussian to 2d
    auto cov2d_culled = projectCov3dTo2d(gaus_centers_cam_frame.index({culled_gaus_ids}), cam, cov3d_culled);
    // 5)  Compute colors via spherical harmonics
    auto colors_culled = shToColor(shs_culled, pws_culled, cam.tcw);
    // 6) Find 3 sigma areas gaussian would cover in image space
    torch::Tensor areas_culled;
    auto          cinv_culled = inverseCov2d(cov2d_culled, areas_culled);
    // 7) Splat gaussians to image space
    auto depths_culled = gaus_centers_cam_frame.index({culled_gaus_ids, 2});
    // auto image         = splat(
    //     cam, gaus_centers_img_frame_culled, cinv_culled, alphas_culled, depths_culled, colors_culled, areas_culled);
    auto image = splatTiled(
        cam, gaus_centers_img_frame_culled, cinv_culled, alphas_culled, depths_culled, colors_culled, areas_culled);
    // auto image = splatTiledBatched(
    // cam, gaus_centers_img_frame_culled, cinv_culled, alphas_culled, depths_culled, colors_culled, areas_culled);

    return image;
}

} // namespace gsplat