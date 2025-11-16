
#include "spherical_harmonics_coefs.h"
#include "typedefs.h"
#include "utils.hpp"
#include <execution>
#include <torch/torch.h>

namespace
{
torch::Tensor upperTriangular(const torch::Tensor &mat)
{
    using torch::indexing::Slice;
    const auto size{mat.size(1)};
    const auto idx = torch::triu_indices(size, size);
    return mat.index({Slice(), idx[0], idx[1]});
}
} // namespace

namespace gsplat
{
using torch::indexing::Slice;
using torch::indexing::TensorIndex;

torch::Tensor GLOBAL_IDX_MAP;

// Projects 3d gaussians from world coordinates to camera coordinates.
// Also returns the image projection of the center of 3d gaussian
torch::Tensor project(torch::Tensor &points_world, const Camera &cam, torch::Tensor &proj_us_out)
{
    const torch::Tensor points_cam = torch::matmul(points_world, cam.Rcw.transpose(0, 1)) + cam.tcw;

    const torch::Tensor x = points_cam.index({torch::indexing::Slice(), 0});
    const torch::Tensor y = points_cam.index({torch::indexing::Slice(), 1});
    const torch::Tensor z = points_cam.index({torch::indexing::Slice(), 2});

    // u = fx * x / z + cx
    // v = fy * y / z + cy
    const torch::Tensor u_x = x.mul(cam.fx).div(z).add(cam.cx);
    const torch::Tensor u_y = y.mul(cam.fy).div(z).add(cam.cy);

    proj_us_out = torch::stack({u_x, u_y}, /*dim=*/1);

    return points_cam;
}

torch::Tensor computeCov3d(const torch::Tensor &scale, const torch::Tensor &quat)
{
    const int64_t batch  = scale.size(0);
    auto const    device = scale.device();
    auto const    dtype  = scale.scalar_type();

    // Build diagonal scale matrix S: shape [batch, 3, 3]
    auto S = torch::zeros({batch, 3, 3}, torch::TensorOptions().device(device).dtype(dtype));
    S.index_put_({Slice(), 0, 0}, scale.index({Slice(), 0}));
    S.index_put_({Slice(), 1, 1}, scale.index({Slice(), 1}));
    S.index_put_({Slice(), 2, 2}, scale.index({Slice(), 2}));

    auto const R{quatToRotMatrix(quat)};

    // M = R @ S, Sigma = M @ M^T
    auto M     = torch::bmm(R, S);
    auto Sigma = torch::bmm(M, M.transpose(1, 2));

    return Sigma;
}

torch::Tensor projectCov3dTo2d(const torch::Tensor &points_cam, const Camera &cam, const torch::Tensor &cov3d)
{
    auto x = points_cam.index({Slice(), 0}); // [batch]
    auto y = points_cam.index({Slice(), 1}); // [batch]
    auto z = points_cam.index({Slice(), 2}); // [batch]

    // Compute field‐of‐view limits
    const float tan_fovx = 2.0F * std::atan(cam.width / (2.0f * cam.fx));
    const float tan_fovy = 2.0F * std::atan(cam.height / (2.0f * cam.fy));
    const float limx     = 1.3F * tan_fovx;
    const float limy     = 1.3F * tan_fovy;

    auto inv_z_x = x.div(z);
    inv_z_x      = torch::clamp(inv_z_x, -limx, limx);
    x            = inv_z_x.mul(z);
    auto inv_z_y = y.div(z);
    inv_z_y      = torch::clamp(inv_z_y, -limy, limy);
    y            = inv_z_y.mul(z);

    const int64_t batch  = points_cam.size(0);
    const auto    device = points_cam.device();
    const auto    dtype  = points_cam.scalar_type();

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

    auto const T     = torch::matmul(J, cam.Rcw); // [batch, 3, 3]
    auto const Sigma = cov3d;
    // Sigma' = T @ Sigma @ Tᵀ
    auto const TSigma      = torch::bmm(T, Sigma);
    auto       Sigma_prime = torch::bmm(TSigma, T.transpose(1, 2));

    // inflate x & y variance by 0.3
    {
        constexpr float kInflation = 0.3F;
        auto const      diag00     = Sigma_prime.index({Slice(), 0, 0});
        Sigma_prime.index_put_({Slice(), 0, 0}, diag00 + kInflation);
        auto const diag11 = Sigma_prime.index({Slice(), 1, 1});
        Sigma_prime.index_put_({Slice(), 1, 1}, diag11 + kInflation);
    }

    // Extract the top‐left 2×2 block:
    auto const Sigma2d = Sigma_prime.index({Slice(), Slice(0, 2), Slice(0, 2)});
    return upperTriangular(Sigma2d);
}

torch::Tensor shToColor(const torch::Tensor &sh, // [batch, sh_dim]
                        const torch::Tensor &pw, // [batch, 3]
                        const torch::Tensor &twc // [batch, 3]
)
{
    const int64_t sh_dim = sh.size(1);

    auto color = SH_C0_0 * sh.index({Slice(), Slice(0, 3)}) + 0.5;
    if (sh_dim <= 3)
    {
        return color;
    }

    auto ray_dir = pw - twc; // [batch, 3]
    ray_dir      = ray_dir / ray_dir.norm(2, /*dim=*/1, /*keepdim=*/true);

    const auto x = ray_dir.index({Slice(), Slice(0, 1)}); // [batch,1]
    const auto y = ray_dir.index({Slice(), Slice(1, 2)}); // [batch,1]
    const auto z = ray_dir.index({Slice(), Slice(2, 3)}); // [batch,1]

    color = color + SH_C1_0 * (y * sh.index({Slice(), Slice(3, 6)})) +
            SH_C1_1 * (z * sh.index({Slice(), Slice(6, 9)})) + SH_C1_2 * (x * sh.index({Slice(), Slice(9, 12)}));
    if (sh_dim <= 12)
    {
        return color;
    }

    // compute second‐order terms
    const auto xx = x * x; // [batch,1]
    const auto yy = y * y;
    const auto zz = z * z;
    const auto xy = x * y;
    const auto yz = y * z;
    const auto xz = x * z;

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
    auto const cov_xx = cov2d.index({Slice(), 0}); // [batch]
    auto const cov_xy = cov2d.index({Slice(), 1}); // [batch]
    auto const cov_yy = cov2d.index({Slice(), 2}); // [batch]

    auto const det_term = cov_xx * cov_yy - cov_xy * cov_xy + 1e-6;
    auto const det_inv  = det_term.reciprocal(); // [batch]

    auto const c00       = cov_yy.mul(det_inv);
    auto const c01       = cov_xy.mul(det_inv).neg();
    auto const c11       = cov_xx.mul(det_inv);
    auto const cov2d_inv = torch::stack({c00, c01, c11}, /*dim=*/1); // [batch, 3]

    const auto stacked = torch::stack({cov_xx, cov_yy}, /*dim=*/1); // [batch, 2]
    const auto sqrted  = torch::sqrt(stacked);
    areas_3_sigma      = sqrted.mul(3.0).to(torch::kInt32); // [batch, 2]

    return cov2d_inv;
}

torch::Tensor splatTiled(const Camera  &cam,
                         torch::Tensor &gaussian_2d_centers,
                         torch::Tensor &cov2d_inv,
                         torch::Tensor &alphas,
                         torch::Tensor &depths,
                         torch::Tensor &colors,
                         torch::Tensor &areas_3_sigma)
{
    const auto device = gaussian_2d_centers.device();
    const auto dtype  = gaussian_2d_centers.dtype();

    // Output RGB
    torch::Tensor image = torch::zeros({cam.height, cam.width, 3}, torch::TensorOptions().device(device).dtype(dtype));

    // Sort indices by depth (ascending)
    torch::Tensor const depth_sorted_idxs = depths.argsort();
    const int64_t       num_gaussians     = depths.size(0);

    // constexpr int TW = 64, TH = 64; // tile width & height
    constexpr int TW = 32, TH = 32; // tile width & height
    const int     n_tiles_x = (cam.width + TW - 1) / TW;
    const int     n_tiles_y = (cam.height + TH - 1) / TH;

    // Precompute which gaussians belong to which tiles
    std::vector<std::vector<int32_t>> gaussians_in_tiles(n_tiles_x * n_tiles_y);
    // Copy data needed for precomputation to CPU
    auto const depth_sorted_idxs_cpu   = depth_sorted_idxs.to(torch::kCPU);
    const auto gaussian_2d_centers_cpu = gaussian_2d_centers.to(torch::kCPU);
    const auto areas_3_sigma_cpu       = areas_3_sigma.to(torch::kCPU);

    const auto *depth_sorted_idxs_ptr   = depth_sorted_idxs_cpu.data_ptr<int64_t>();
    const auto *gaussian_2d_centers_ptr = gaussian_2d_centers_cpu.data_ptr<float>();
    const auto *areas_3_sigma_ptr       = areas_3_sigma_cpu.data_ptr<int32_t>();
    for (int64_t k = 0; k < num_gaussians; ++k)
    {
        // In each tile, should iterate the gaussians based on depth priority
        const auto i = depth_sorted_idxs_ptr[k];

        const float gaus_center_x = gaussian_2d_centers_ptr[i * 2 + 0];
        const float gaus_center_y = gaussian_2d_centers_ptr[i * 2 + 1];
        const int   rx            = areas_3_sigma_ptr[i * 2 + 0];
        const int   ry            = areas_3_sigma_ptr[i * 2 + 1];

        const int gaus_rect_x0 = std::max(0, static_cast<int>(std::floor(gaus_center_x - rx)));
        const int gaus_rect_x1 = std::min(static_cast<int>(cam.width), static_cast<int>(std::ceil(gaus_center_x + rx)));
        const int gaus_rect_y0 = std::max(0, static_cast<int>(std::floor(gaus_center_y - ry)));
        const int gaus_rect_y1 =
            std::min(static_cast<int>(cam.height), static_cast<int>(std::ceil(gaus_center_y + ry)));

        if ((gaus_rect_x1 <= gaus_rect_x0) || (gaus_rect_y1 <= gaus_rect_y0))
            continue;

        // Assing the gaussian to the tiles it overlaps with
        const int tx0 = std::clamp(gaus_rect_x0 / TW, 0, n_tiles_x - 1);
        const int tx1 = std::clamp((gaus_rect_x1 - 1) / TW, 0, n_tiles_x - 1);
        const int ty0 = std::clamp(gaus_rect_y0 / TH, 0, n_tiles_y - 1);
        const int ty1 = std::clamp((gaus_rect_y1 - 1) / TH, 0, n_tiles_y - 1);

        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx)
                gaussians_in_tiles[ty * n_tiles_x + tx].push_back(static_cast<int32_t>(i));
    }

    // Main splatting logic per each tile
    // x0t, x1t, y0t, y1t: tile bounds in pixel coordinates | tx, ty: tile indices
    auto splat_tile = [&](const int x0t, const int x1t, const int y0t, const int y1t, const int tx, const int ty)
    {
        const int num_pixels_in_tile = (y1t - y0t) * (x1t - x0t);
        const int tile_height        = y1t - y0t;
        const int tile_width         = x1t - x0t;

        auto &gaus_ids_in_this_tile_vec = gaussians_in_tiles[ty * n_tiles_x + tx];
        if (gaus_ids_in_this_tile_vec.empty())
            return;

        // Convert indices to Tensor
        torch::Tensor gaus_ids_in_this_tile = torch::from_blob(gaus_ids_in_this_tile_vec.data(),
                                                               {static_cast<int64_t>(gaus_ids_in_this_tile_vec.size())},
                                                               torch::TensorOptions().dtype(torch::kInt32))
                                                  .to(device)
                                                  .to(torch::kLong)
                                                  .clone();

        // Gather Gaussian attributes (NOTE: index_select copies data)
        torch::Tensor means2D               = gaussian_2d_centers.index_select(0, gaus_ids_in_this_tile);
        torch::Tensor cinv                  = cov2d_inv.index_select(0, gaus_ids_in_this_tile);
        torch::Tensor opacity               = alphas.index_select(0, gaus_ids_in_this_tile);
        torch::Tensor colors_tile           = colors.index_select(0, gaus_ids_in_this_tile);
        const int64_t num_gaussians_in_tile = gaus_ids_in_this_tile.size(0);

        // Pixel grid for this tile
        // [2, tile_height, tile_width]
        auto sub_region = GLOBAL_IDX_MAP.index({Slice(), Slice(y0t, y1t), Slice(x0t, x1t)});
        // Reshape to [num_pixels_in_tile, 2]
        torch::Tensor tile_xy = sub_region.permute({1, 2, 0}).reshape({num_pixels_in_tile, 2}).to(dtype);

        // Gaussian's weight per pixel: Difference between the tile coordinates and the gaussian centers
        // [num_pixels_in_tile, num_gaussians_in_tile] = [num_pixels_in_tile, 1] - [1, num_gaussians_in_tile]
        torch::Tensor dx = tile_xy.index({Slice(), 0}).unsqueeze(1) - means2D.index({Slice(), 0}).unsqueeze(0);
        torch::Tensor dy = tile_xy.index({Slice(), 1}).unsqueeze(1) - means2D.index({Slice(), 1}).unsqueeze(0);

        torch::Tensor c00 = cinv.index({Slice(), 0}).unsqueeze(0);
        torch::Tensor c01 = cinv.index({Slice(), 1}).unsqueeze(0);
        torch::Tensor c11 = cinv.index({Slice(), 2}).unsqueeze(0);

        // The Gaussian density for a pixel at offset x=(dx,dy) is:
        // w = exp(-0.5 * x^t * Cinv * x)
        // x^t * Cinv * x = dx^2 * c00 + dy^2 * c11 + 2*dx*dy*c01
        // [num_pixels_in_tile, num_gaussians_in_tile]
        torch::Tensor quad = dx.mul(dx).mul(c00) + dy.mul(dy).mul(c11) + dx.mul(dy).mul(2.0).mul(c01);

        // α = exp(-0.5*quad) * opacity
        torch::Tensor opacity_view      = opacity.view({1, num_gaussians_in_tile});
        torch::Tensor opacity_broadcast = opacity_view.expand({num_pixels_in_tile, num_gaussians_in_tile});
        // [num_pixels_in_tile, num_gaussians_in_tile]
        torch::Tensor alpha = torch::exp(-0.5 * quad).mul(opacity_broadcast).clamp_max(0.99);

        // Calculate exclusive transmittence: Each gaussian's color contribution is weighted by
        // transmittence of all the layers in front of it, but not itself
        // T_j = prod_{k<j} (1 - α_k)

        // T_before_roll = [[t0, t1, t2, t3]]
        // T_after_roll = [[t3, t0, t1, t2]]
        // T_final = [[1, t0, t1, t2]]
        torch::Tensor one_minus = 1.0 - alpha;                  // [num_pixels_in_tile, num_gaussians_in_tile]
        torch::Tensor T         = torch::cumprod(one_minus, 1); // inclusive transmittence
        T                       = torch::roll(T, /*shifts=*/{1}, /*dims=*/{1}); // circular shift right by 1
        T.index_put_({Slice(), 0}, 1.0);                                        // fix first col to 1 for exclusivity
        // T = [num_pixels_in_tile, num_gaussians_in_tile]

        // Color accumulation
        torch::Tensor weights  = T * alpha;                           // [num_pixels_in_tile, num_gaussians_in_tile]
        torch::Tensor tile_col = torch::matmul(weights, colors_tile); // [num_pixels_in_tile,3]

        // Write back
        image.index_put_({Slice(y0t, y1t), Slice(x0t, x1t), Slice()}, tile_col.view({tile_height, tile_width, 3}));
    };

    // ----- SERIAL VERSION -----
    // Run splatting for each tile
    for (int ty = 0; ty < n_tiles_y; ++ty)
    {
        const int y0t = ty * TH;
        const int y1t = std::min(static_cast<int>(cam.height), y0t + TH);

        for (int tx = 0; tx < n_tiles_x; ++tx)
        {
            const int x0t = tx * TW;
            const int x1t = std::min(static_cast<int>(cam.width), x0t + TW);

            splat_tile(x0t, x1t, y0t, y1t, tx, ty);
        }
    }

    // ----- PARALLEL VERSION ----- (slower due to thread launch overhead)
    // std::vector<std::pair<int, int>> tiles;
    // tiles.reserve(n_tiles_x * n_tiles_y);
    // for (int ty = 0; ty < n_tiles_y; ++ty)
    //     for (int tx = 0; tx < n_tiles_x; ++tx)
    //         tiles.emplace_back(tx, ty);

    // std::for_each(std::execution::par,
    //               tiles.begin(),
    //               tiles.end(),
    //               [&](auto const tile)
    //               {
    //                   const int tx = tile.first;
    //                   const int ty = tile.second;

    //                   const int y0t = ty * TH;
    //                   const int y1t = std::min(static_cast<int>(cam.height), y0t + TH);
    //                   const int x0t = tx * TW;
    //                   const int x1t = std::min(static_cast<int>(cam.width), x0t + TW);

    //                   splat_tile(x0t, x1t, y0t, y1t, tx, ty);
    //               });

    return image;
}

torch::Tensor forwardWithCulling(LearnableParams &params,
                                 const Camera    &cam,
                                 torch::Tensor   &depths_in_cam_frame_out,
                                 torch::Tensor   &gaus_centers_img_frame_out)
{
    auto alphas = getAlphas(params.alphas_raw);
    auto scales = getScales(params.scales_raw);
    auto rots   = getRots(params.rots_raw);

    // 1) Project 3D gaussian centers to camera coordinates and image coordinates
    auto gaus_centers_cam_frame = project(params.pws, cam, gaus_centers_img_frame_out);
    // Normally not part of the computational graph, however needed for gradient accumulation
    // in image space, which is later used in adaptive densification
    gaus_centers_img_frame_out.retain_grad();

    // 2) Build frustum mask and filter out gaussians outside the frustum
    const auto z = gaus_centers_cam_frame.index({Slice(), 2}); // [N]
    // Calculate normalized device coordinates of the gaussian centers in the image frame
    const auto u_ndc    = gaus_centers_img_frame_out.index({Slice(), 0}) / cam.width * 2.F - 1.F;  // map to [-1, 1]
    const auto v_ndc    = gaus_centers_img_frame_out.index({Slice(), 1}) / cam.height * 2.F - 1.F; // map to [-1, 1]
    const auto radii_3d = std::get<0>(scales.max(1, false)) * 3.0f;   // 3 sigma based on gaussian covariance
    const auto r_x_ndc  = radii_3d * (cam.fx / z) / cam.width * 2.0f; // radius in normalized image coordinates
    const auto r_y_ndc  = radii_3d * (cam.fy / z) / cam.height * 2.0f;
    // Check if within the camera frustum (6 sides)
    constexpr float kNearClip    = 0.2F;
    constexpr float kFarClip     = 100.0F;
    auto const frustum_cull_mask = (z > kNearClip) & (z < kFarClip) & (u_ndc + r_x_ndc > -1) & (u_ndc - r_x_ndc < 1) &
                                   (v_ndc + r_y_ndc > -1) & (v_ndc - r_y_ndc < 1);
    auto const culled_gaus_ids = frustum_cull_mask.nonzero().squeeze(1); // [M]
    std::cout << "Survived Gaussians after Frustum Culling: " << culled_gaus_ids.size(0) << " / " << params.pws.size(0)
              << std::endl;

    // Pick the parameters of the surviving Gaussians (causes copy)
    auto pws_culled                    = params.pws.index({culled_gaus_ids});                 // [M,3]
    auto alphas_culled                 = alphas.index({culled_gaus_ids});                     // [M]
    auto scales_culled                 = scales.index({culled_gaus_ids});                     // [M,3]
    auto rots_culled                   = rots.index({culled_gaus_ids});                       // [M,3,3]
    auto gaus_centers_img_frame_culled = gaus_centers_img_frame_out.index({culled_gaus_ids}); // [M,2]
    auto shs_culled                    = params.shs.index({culled_gaus_ids});                 // [M, sh_dim]

    // 3) Compute 3D covariance matrices for each gaussian
    auto const cov3d_culled = computeCov3d(scales_culled, rots_culled);
    // 4) Project 3d gaussian to 2d
    auto const cov2d_culled = projectCov3dTo2d(gaus_centers_cam_frame.index({culled_gaus_ids}), cam, cov3d_culled);
    // 5)  Compute colors via spherical harmonics
    auto colors_culled = shToColor(shs_culled, pws_culled, cam.tcw);
    // 6) Find 3 sigma areas gaussian would cover in image space
    torch::Tensor areas_culled;
    auto          cinv_culled = inverseCov2d(cov2d_culled, areas_culled);
    // 7) Splat gaussians to image space
    auto depths_culled = gaus_centers_cam_frame.index({culled_gaus_ids, 2});

    auto image = splatTiled(
        cam, gaus_centers_img_frame_culled, cinv_culled, alphas_culled, depths_culled, colors_culled, areas_culled);

    depths_in_cam_frame_out = z;
    return image;
}

} // namespace gsplat