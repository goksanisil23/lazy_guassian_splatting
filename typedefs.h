#pragma once
#include <torch/torch.h>

namespace gsplat
{
struct Vec3d
{
    float x, y, z;
};

struct Vec4d
{
    float x, y, z, w;
};

struct Camera
{
    int32_t       id;
    int64_t       width, height;
    double        fx, fy, cx, cy; // assuming pinhole model
    torch::Tensor Rcw;
    torch::Tensor tcw; // camera center in world coordinates
    torch::Tensor twc;
    std::string   image_path;
};

struct Gaussians
{
    std::vector<Vec3d> pws;    // world coordinates of the Gaussian centers
    std::vector<Vec3d> shs;    // spherical harmonics coefficients
    std::vector<Vec3d> scales; // scales
    std::vector<Vec4d> rots;   // RGB colors of the Gaussians
    std::vector<float> alphas; // alpha values for transparency
};
} // namespace gsplat

struct LearnableParams
{
    torch::Tensor pws;
    torch::Tensor low_shs;
    torch::Tensor high_shs;
    torch::Tensor alphas_raw;
    torch::Tensor scales_raw;
    torch::Tensor rots_raw;
};

struct AdamsParams
{
    const float pws_lr        = 0.001f;
    const float low_shs_lr    = 0.001f;
    const float high_shs_lr   = 0.001f / 20.f;
    const float alphas_raw_lr = 0.05f;
    const float scales_raw_lr = 0.005f;
    const float rots_raw_lr   = 0.001f;
};