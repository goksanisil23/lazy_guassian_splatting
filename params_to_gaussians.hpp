#pragma once
#include <torch/torch.h>

#include "typedefs.h"
#include "utils.hpp"

LearnableParams createLearningParamsFromGaussians(const gsplat::Gaussians &gaussians,
                                                  const torch::Device     &device,
                                                  const bool               use_higher_shs)
{
    LearnableParams params;
    // 1) pws
    params.pws = torch::from_blob((void *)(gaussians.pws.data()),
                                  {static_cast<int64_t>(gaussians.pws.size()), 3},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .to(device)
                     .set_requires_grad(true); // [N,3]

    // 2) rots_raw
    params.rots_raw = torch::from_blob((void *)(gaussians.rots.data()),
                                       {static_cast<int64_t>(gaussians.rots.size()), 4},
                                       torch::TensorOptions().dtype(torch::kFloat32))
                          .to(device)
                          .set_requires_grad(true); // [N,4]

    // 3) scales_raw
    params.scales_raw = getScalesRaw(torch::from_blob((void *)(gaussians.scales.data()),
                                                      {static_cast<int64_t>(gaussians.scales.size()), 3},
                                                      torch::TensorOptions().dtype(torch::kFloat32))
                                         .to(device))
                            .set_requires_grad(true); // [N,3]

    // 4) alphas_raw
    params.alphas_raw = getAlphasRaw(torch::from_blob((void *)(gaussians.alphas.data()),
                                                      {static_cast<int64_t>(gaussians.alphas.size()), 1},
                                                      torch::TensorOptions().dtype(torch::kFloat32))
                                         .to(device))
                            .set_requires_grad(true); // [N,1]

    // 5) low and optional high shs
    auto low_shs = torch::from_blob((void *)(gaussians.shs.data()),
                                    {static_cast<int64_t>(gaussians.shs.size()), 3},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .to(device); // [N,3]
    // concatanate and make differentiable
    if (use_higher_shs)
    {
        auto high_shs = torch::ones_like(low_shs).repeat({1, 15}).mul(0.001f).to(device); // [N,45]
        params.shs    = torch::cat({low_shs.detach(), high_shs.detach()}, 1).detach().set_requires_grad(true);
    }
    else
    {
        params.shs = low_shs.detach().set_requires_grad(true);
    }

    return params;
}

gsplat::Gaussians writeParamsToGaussians(const LearnableParams &params)
{
    using namespace gsplat;
    Gaussians gaussians;

    auto pws      = params.pws.detach().to(torch::kCPU);
    auto rots_raw = params.rots_raw.detach().to(torch::kCPU);
    auto scales_r = params.scales_raw.detach().to(torch::kCPU);
    auto alphas_r = params.alphas_raw.detach().to(torch::kCPU);
    auto shs      = params.shs.detach().to(torch::kCPU);

    const uint32_t num_gaussians{static_cast<uint32_t>(pws.size(0))};

    gaussians.pws.resize(num_gaussians);
    gaussians.rots.resize(num_gaussians);
    gaussians.scales.resize(num_gaussians);
    gaussians.alphas.resize(num_gaussians);
    gaussians.shs.resize(num_gaussians);

    // assuming contiguous tensors and Vec3d/Vec4d are POD float/double types
    auto pws_ptr      = pws.data_ptr<float>(); // or double if that’s your real type
    auto rots_ptr     = rots_raw.data_ptr<float>();
    auto scales_r_ptr = scales_r.data_ptr<float>();
    auto alphas_r_ptr = alphas_r.data_ptr<float>();
    auto shs_ptr      = shs.data_ptr<float>();

    for (uint32_t i = 0; i < num_gaussians; ++i)
    {
        gaussians.pws[i]    = Vec3d{pws_ptr[3 * i + 0], pws_ptr[3 * i + 1], pws_ptr[3 * i + 2]};
        gaussians.rots[i]   = Vec4d{rots_ptr[4 * i + 0], rots_ptr[4 * i + 1], rots_ptr[4 * i + 2], rots_ptr[4 * i + 3]};
        auto const scales   = getScales(scales_r[i]);
        gaussians.scales[i] = Vec3d{scales[0].item<float>(), scales[1].item<float>(), scales[2].item<float>()};
        auto const alpha    = getAlphas(alphas_r[i]);
        gaussians.alphas[i] = alpha.item<float>();

        // Not saving the higher order terms, even if they're trained
        std::vector<float> sh_dc_term(3);
        sh_dc_term[0] = shs_ptr[3 * i + 0];
        sh_dc_term[1] = shs_ptr[3 * i + 1];
        sh_dc_term[2] = shs_ptr[3 * i + 2];

        gaussians.shs[i] = Vec3d{sh_dc_term[0], sh_dc_term[1], sh_dc_term[2]};
    }

    return gaussians;
}