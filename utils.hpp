#pragma once
#include "typedefs.h"
#include <torch/torch.h>

// inverse of sigmoid: log(x / (1 - x))
inline at::Tensor getAlphasRaw(const at::Tensor &x)
{
    return torch::log(x / (1 - x));
}

inline float getAlphasRaw(const float x)
{
    return std::log(x / (1 - x));
}

inline at::Tensor getScalesRaw(const at::Tensor &x)
{
    return torch::log(x);
}

inline float getScalesRaw(const float x)
{
    return std::log(x);
}

inline at::Tensor getAlphas(const at::Tensor &x)
{
    return torch::sigmoid(x);
}

inline at::Tensor getScales(const at::Tensor &x)
{
    return torch::exp(x);
}

inline at::Tensor getRots(const at::Tensor &x)
{
    return torch::nn::functional::normalize(x);
}

inline at::Tensor getShs(const torch::Tensor &low_shs, const torch::Tensor &high_shs)
{
    return torch::cat(std::vector<torch::Tensor>{low_shs, high_shs}, /*dim=*/1);
}

torch::Tensor makeIdxMap(int H, int W, const torch::Device &device)
{
    auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
    auto ys      = torch::arange(H, options);
    auto xs      = torch::arange(W, options);
    auto grids   = torch::meshgrid({ys, xs}); // default “ij” ordering
    // stack {x‐grid, y‐grid} to get [2,H,W] with [0,:,:]=x, [1,:,:]=y
    return torch::stack({grids[1], grids[0]}, /*dim=*/0);
}

std::vector<torch::optim::OptimizerParamGroup> createAdamParamGroup(const LearnableParams &params)
{
    const AdamsParams adams_params;

    std::vector<torch::optim::OptimizerParamGroup> param_groups;

    param_groups.emplace_back(std::vector<torch::Tensor>{params.pws},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.pws_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.shs},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.shs_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.alphas_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.alphas_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.scales_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.scales_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.rots_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.rots_raw_lr));

    return param_groups;
}

torch::Tensor quatToRotMatrix(const torch::Tensor &quat)
{
    using torch::indexing::Slice;

    auto w = quat.index({Slice(), 0});
    auto x = quat.index({Slice(), 1});
    auto y = quat.index({Slice(), 2});
    auto z = quat.index({Slice(), 3});

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

    return R;
}
