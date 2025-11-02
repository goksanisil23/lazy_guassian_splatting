#pragma once
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
    param_groups.emplace_back(std::vector<torch::Tensor>{params.low_shs},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.low_shs_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.high_shs},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.high_shs_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.alphas_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.alphas_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.scales_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.scales_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.rots_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams_params.rots_raw_lr));

    return param_groups;
}
