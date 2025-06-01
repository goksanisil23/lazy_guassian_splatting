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
