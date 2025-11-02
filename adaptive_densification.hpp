#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

#include "typedefs.h"
#include "utils.hpp"

namespace
{
constexpr float kAlphaThresh              = 0.005F;
constexpr float kGradThresh               = 4e-7;
constexpr float kScaleThreshPruning       = 0.1F;
constexpr float kScaleThreshDensification = 0.01F;

void pruneParams(LearnableParams &params, const torch::Tensor &prune_mask, const torch::Tensor &keep_mask)
{
    auto replace_in_place = [](torch::Tensor &src_tensor, const torch::Tensor &mask)
    { src_tensor = src_tensor.index({mask}).detach().set_requires_grad(true); };

    std::cout << "Pruned Gaussians: " << prune_mask.sum().item<int64_t>() << " / " << params.pws.size(0) << std::endl;
    replace_in_place(params.pws, keep_mask);
    replace_in_place(params.low_shs, keep_mask);
    replace_in_place(params.high_shs, keep_mask);
    replace_in_place(params.alphas_raw, keep_mask);
    replace_in_place(params.scales_raw, keep_mask);
    replace_in_place(params.rots_raw, keep_mask);
}

} // namespace

void updateAccumulatedGradInfo(GradAccumInfo &grad_accum_info, const torch::Tensor &param, const torch::Tensor &mask)
{
    // Gradient of the loss function w.r.t. the parameter
    auto const d_loss_d_param = param.grad();

    if (!d_loss_d_param.defined())
        throw std::runtime_error("Gradient is not defined for the parameter!");

    // Norm of the 2d gaussian center gradients
    auto const grad_norm = torch::linalg_vector_norm(d_loss_d_param, 2, /*dim=*/1, /*keepdim=*/true); // [N,1]

    torch::NoGradGuard no_grad;

    if (!grad_accum_info.count.defined()) // 1st iteration
    {
        // Initialize
        grad_accum_info.count       = mask.to(torch::kInt32);
        grad_accum_info.accum_grads = grad_norm.clone();
    }
    else
    {
        // Accumulate
        grad_accum_info.count += mask;
        grad_accum_info.accum_grads.index_put_({mask},
                                               grad_accum_info.accum_grads.index({mask}) + grad_norm.index({mask}));
    }
}

void adaptiveDensification(LearnableParams     &params,
                           torch::optim::Adam  &optimizer,
                           const GradAccumInfo &grad_accum_info,
                           const float          scene_scale)
{
    using torch::indexing::Slice;
    auto const device = params.pws.device();

    // ------ 1) Pruning ------ //

    const float scale_thresh_pruning = kScaleThreshPruning * scene_scale;

    auto const alpha_thresh_raw  = getAlphasRaw(kAlphaThresh);
    auto const scales_raw_thresh = getScalesRaw(scale_thresh_pruning);

    auto const small_alpha_prune_mask  = params.alphas_raw.squeeze().lt(alpha_thresh_raw);
    auto const [max_vals, max_indices] = torch::max(params.scales_raw, 1);
    auto const big_scale_prune_mask    = max_vals.gt(scales_raw_thresh);

    auto const prune_mask = torch::logical_or(small_alpha_prune_mask, big_scale_prune_mask);
    auto const keep_mask  = torch::logical_not(prune_mask);

    pruneParams(params, prune_mask, keep_mask);

    auto pws      = params.pws;
    auto low_shs  = params.low_shs;
    auto high_shs = params.high_shs;
    auto alphas   = getAlphas(params.alphas_raw);
    auto scales   = getScales(params.scales_raw);
    auto rots     = getRots(params.rots_raw);

    // ------ 2) Densification ------ //
    // Calculate the average

    const float scale_thresh_densification = kScaleThreshDensification * scene_scale;

    auto const avg_grads =
        grad_accum_info.accum_grads.squeeze().index({keep_mask}).div(grad_accum_info.count.index({keep_mask}));
    // replace nans with zeros (since count can be zero)
    avg_grads.nan_to_num_(0.0);

    auto const high_grad_densify_mask    = avg_grads.gt(kGradThresh);
    auto const [max_vals_, max_indices_] = torch::max(scales, 1);
    auto const small_scale_densify_mask  = max_vals_.lt(scale_thresh_densification);

    auto const clone_mask = torch::logical_and(high_grad_densify_mask, small_scale_densify_mask);
    auto const split_mask = torch::logical_and(high_grad_densify_mask, torch::logical_not(small_scale_densify_mask));

    // Clone gaussians
    auto pws_cloned      = pws.index({clone_mask});
    auto low_shs_cloned  = low_shs.index({clone_mask});
    auto high_shs_cloned = high_shs.index({clone_mask});
    auto alphas_cloned   = alphas.index({clone_mask});
    auto scales_cloned   = scales.index({clone_mask});
    auto rots_cloned     = rots.index({clone_mask});

    // // Split gaussians
    // auto const rots_split = rots.index({split_mask});

    auto append_params = [](torch::Tensor &old_params, const torch::Tensor &new_params)
    { return torch::cat({old_params.detach(), new_params.detach()}, 0).set_requires_grad(true); };

    params.pws        = append_params(pws, pws_cloned);
    params.low_shs    = append_params(low_shs, low_shs_cloned);
    params.high_shs   = append_params(high_shs, high_shs_cloned);
    params.alphas_raw = append_params(params.alphas_raw, getAlphasRaw(alphas_cloned));
    params.scales_raw = append_params(params.scales_raw, getScalesRaw(scales_cloned));
    params.rots_raw   = append_params(params.rots_raw, rots_cloned);

    // Rebuild the optimizer
    {
        std::vector<torch::optim::OptimizerParamGroup> param_groups{createAdamParamGroup(params)};

        optimizer.param_groups() = std::move(param_groups);
        optimizer.state().clear(); // optimizer holds momentum associated to the old parameters
        optimizer.zero_grad();     // old gradients associated to old parameters might have stale shapes
    }
}
