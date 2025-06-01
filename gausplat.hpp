
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

    // Sort indices by depth (ascending)
    torch::Tensor depth_sorted_idxs = depths.argsort();
    int64_t       num_gaussians     = depths.size(0);

    for (int64_t k = 0; k < num_gaussians; ++k)
    {
        std::cout << "Processing gaussian " << k + 1 << " / " << num_gaussians << "\r";

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
        float gx = gx_tensor.item<float>();
        float gy = gy_tensor.item<float>();

        // Quickly cull if normalized center is out of ±1.3 range
        if (std::abs(gx / cam.width) > 1.3f || std::abs(gy / cam.height) > 1.3f)
        {
            continue;
        }

        // Fetch the integer 3D-sigma “radius” in pixels: [r_x, r_y]
        int r_x = areas_3_sigma.index({idx, 0}).item<int>();
        int r_y = areas_3_sigma.index({idx, 1}).item<int>();

        // Compute bounding box [x0,x1) × [y0,y1) in image space, clamped
        float x0f = gx - r_x;
        float x1f = gx + r_x;
        float y0f = gy - r_y;
        float y1f = gy + r_y;

        int x0 = static_cast<int>(std::clamp(x0f, 0.0f, float(cam.width)));
        int x1 = static_cast<int>(std::clamp(x1f, 0.0f, float(cam.width)));
        int y0 = static_cast<int>(std::clamp(y0f, 0.0f, float(cam.height)));
        int y1 = static_cast<int>(std::clamp(y1f, 0.0f, float(cam.height)));

        if (x1 <= x0 || y1 <= y0)
        {
            continue; // invalid patch
        }

        // Slice out the sub-region from GLOBAL_IDX_MAP: shape [2, dy, dx]
        auto sub   = GLOBAL_IDX_MAP.index({
            Slice(),       // both channels
            Slice(y0, y1), // rows
            Slice(x0, x1)  // cols
        });
        auto sub_x = sub.index({0}); // [patch_h, patch_w], X‐coordinates
        auto sub_y = sub.index({1}); // [patch_h, patch_w], Y‐coordinates

        // Pull out inverse-covariance scalars for this pixel: [c00, c01, c11]
        auto cinv00 = cov2d_inv.index({idx, 0}); // 0-d Tensor
        auto cinv01 = cov2d_inv.index({idx, 1}); // 0-d Tensor
        auto cinv11 = cov2d_inv.index({idx, 2}); // 0-d Tensor

        // Compute Mahalanobis distance using Tensors:
        // d0 = (x - gx), d1 = (y - gy)
        auto d0   = sub_x - gx_tensor; // [patch_h, patch_w], still in graph
        auto d1   = sub_y - gy_tensor; // [patch_h, patch_w]
        auto maha = cinv00 * (d0 * d0) + cinv11 * (d1 * d1) + 2.0f * cinv01 * (d0 * d1); // [patch_h, patch_w]

        auto alpha_i     = alphas.index({idx});
        auto patch_alpha = torch::exp(-0.5f * maha).mul(alpha_i).clamp_max(0.99f); // [dy,dx]
        // auto color_i     = colors.index({idx});
        auto color_i = colors.index({idx}).view({1, 1, 3});
        auto patch   = patch_alpha.unsqueeze(-1).mul(color_i);

        // — Write back into image in-place:
        // image[y0:y1, x0:x1, :] += patch
        image.index_put_({Slice(y0, y1), Slice(x0, x1), Slice()},
                         image.index({Slice(y0, y1), Slice(x0, x1), Slice()}) + patch);
    }
    std::cout << std::endl; // flush the progress bar

    return image;
}

torch::Tensor forward(LearnableParams &params, const Camera &cam)
{
    // Alphas are limited to [0, 1] range, via sigmoid
    auto alphas = getAlphas(params.alphas_raw);
    // Scales are limited to > 0, via exp
    auto scales = getScales(params.scales_raw);
    // Rots norms are limited to 1, via normalization
    auto rots = getRots(params.rots_raw);

    auto shs = getShs(params.low_shs, params.high_shs);

    // 1) Project 3D gaussian centers to camera coordinates and image coordinates
    torch::Tensor proj_us;
    auto          points_cam = project(params.pws, cam, proj_us);
    auto          us         = proj_us.clone().detach().requires_grad_(true);

    // 2) Compute 3D covariance matrices for each gaussian
    auto cov3d = computeCov3d(scales, rots);

    // 3) Project 3d gaussian to 2d
    auto cov2d = projectCov3dTo2d(points_cam, cam, cov3d);

    // 4) Compute colors via spherical harmonics
    auto colors = shToColor(shs, params.pws, cam.tcw);

    // 5) Find 3 sigma areas gaussian would cover in image space
    torch::Tensor areas_3_sigma;
    auto          cov2d_inv = inverseCov2d(cov2d, areas_3_sigma);

    // 6) Splat gaussians to image space
    auto depths = points_cam.index({Slice(), 2});
    auto image  = splat(cam, us, cov2d_inv, alphas, depths, colors, areas_3_sigma);

    // auto mask = depths > 0.2;

    return image;
}
} // namespace gsplat