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
constexpr float kSplitScaleFactor         = 0.6;

constexpr bool kEnablePruning = false;

void pruneParams(LearnableParams &params, const torch::Tensor &prune_mask, const torch::Tensor &keep_mask)
{
    auto replace_in_place = [](torch::Tensor &src_tensor, const torch::Tensor &mask)
    { src_tensor = src_tensor.index({mask}).detach().set_requires_grad(true); };

    std::cout << "Pruned Gaussians: " << prune_mask.sum().item<int64_t>() << " / " << params.pws.size(0) << std::endl;
    replace_in_place(params.pws, keep_mask);
    replace_in_place(params.shs, keep_mask);
    replace_in_place(params.alphas_raw, keep_mask);
    replace_in_place(params.scales_raw, keep_mask);
    replace_in_place(params.rots_raw, keep_mask);
}

torch::Tensor rotateByQuaternion(const torch::Tensor &quat, const torch::Tensor &vec)
{

    auto const R{quatToRotMatrix(quat)}; // (N,3,3)
    // apply rotation: (N,3,3) @ (N,3,1) -> (N,3)
    return torch::bmm(R, vec.unsqueeze(-1)).squeeze(-1);
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

    auto const device      = params.pws.device();
    auto const dtype       = params.pws.scalar_type();
    auto const tensor_opts = torch::TensorOptions().device(device).dtype(dtype);

    // ------ 1) Pruning ------ //
    torch::Tensor keep_mask;
    if constexpr (kEnablePruning)
    {
        const float scale_thresh_pruning = kScaleThreshPruning * scene_scale;

        auto const alpha_thresh_raw  = getAlphasRaw(kAlphaThresh);
        auto const scales_raw_thresh = getScalesRaw(scale_thresh_pruning);

        auto const small_alpha_prune_mask  = params.alphas_raw.squeeze().lt(alpha_thresh_raw);
        auto const [max_vals, max_indices] = torch::max(params.scales_raw, 1);
        auto const big_scale_prune_mask    = max_vals.gt(scales_raw_thresh);

        std::cout << "Prune masks: " << "  Small alpha count: " << small_alpha_prune_mask.sum().item<int64_t>()
                  << "  Big scale count: " << big_scale_prune_mask.sum().item<int64_t>() << std::endl;

        auto const prune_mask = torch::logical_or(small_alpha_prune_mask, big_scale_prune_mask);
        keep_mask             = torch::logical_not(prune_mask);

        pruneParams(params, prune_mask, keep_mask);
    }
    else
    {
        keep_mask = torch::ones({params.pws.size(0)}, torch::TensorOptions().device(device).dtype(torch::kBool));
    }

    auto pws = params.pws;
    auto shs = params.shs;
    // auto high_shs = params.high_shs;
    auto alphas = getAlphas(params.alphas_raw);
    auto scales = getScales(params.scales_raw);
    auto rots   = getRots(params.rots_raw);

    // ------ 2) Densification ------ //
    const float scale_thresh_densification = kScaleThreshDensification * scene_scale;

    auto const avg_grads =
        grad_accum_info.accum_grads.squeeze().index({keep_mask}).div(grad_accum_info.count.index({keep_mask}));
    // replace nans with zeros (caused by count being zero)
    avg_grads.nan_to_num_(0.0);

    auto const high_grad_densify_mask    = avg_grads.gt(kGradThresh);
    auto const [max_vals_, max_indices_] = torch::max(scales, 1);
    auto const small_scale_densify_mask  = max_vals_.lt(scale_thresh_densification);

    auto const clone_mask = torch::logical_and(high_grad_densify_mask, small_scale_densify_mask);
    auto const split_mask = torch::logical_and(high_grad_densify_mask, torch::logical_not(small_scale_densify_mask));
    auto const split_size = split_mask.sum().item<int64_t>();

    std::cout << "Densify masks: " << "  High grad count: " << high_grad_densify_mask.sum().item<int64_t>()
              << "  Small scale count: " << small_scale_densify_mask.sum().item<int64_t>()
              << "  Clone count: " << clone_mask.sum().item<int64_t>() << "  Split count: " << split_size << std::endl;

    // Clone gaussians
    auto pws_clone = pws.index({clone_mask});
    auto shs_clone = shs.index({clone_mask});
    // auto high_shs_clone = high_shs.index({clone_mask});
    auto alphas_clone = alphas.index({clone_mask});
    auto scales_clone = scales.index({clone_mask});
    auto rots_clone   = rots.index({clone_mask});

    // Split gaussians
    auto const          rots_split               = rots.index({split_mask});
    auto                split_gaussian_rot_means = torch::zeros({split_size, 3}, tensor_opts);
    auto                split_gaussian_rot_stds  = scales.index({split_mask});
    torch::Tensor const samples      = at::normal(split_gaussian_rot_means, split_gaussian_rot_stds, c10::nullopt);
    auto const          pws_split    = pws.index({split_mask}) + rotateByQuaternion(rots_split, samples);
    auto const          alphas_split = alphas.index({split_mask});
    auto const          scales_split = scales.index({split_mask}).mul(kSplitScaleFactor);
    auto const          shs_split    = shs.index({split_mask});
    // auto const          high_shs_split = high_shs.index({split_mask});

    auto append_params =
        [](const torch::Tensor &old_params, const torch::Tensor &new_params_1, const torch::Tensor &new_params_2)
    {
        // cat returns a tensor that is not a leaf, whereas Adam requires leaf tensors
        // Therefore, rebuild concatanated tensors as leaf tensors
        auto out = torch::cat({old_params.detach(), new_params_1.detach(), new_params_2.detach()}, 0);
        return out.detach().set_requires_grad(true);
    };

    params.pws = append_params(pws, pws_clone, pws_split);
    params.shs = append_params(shs, shs_clone, shs_split);
    // params.high_shs   = append_params(high_shs, high_shs_clone, high_shs_split);
    params.alphas_raw = append_params(params.alphas_raw, getAlphasRaw(alphas_clone), getAlphasRaw(alphas_split));
    params.scales_raw = append_params(params.scales_raw, getScalesRaw(scales_clone), getScalesRaw(scales_split));
    params.rots_raw   = append_params(params.rots_raw, rots_clone, rots_split);

    // Rebuild the optimizer with new parameters (no momentum saving)
    {
        std::vector<torch::optim::OptimizerParamGroup> param_groups{createAdamParamGroup(params)};

        optimizer.param_groups() = std::move(param_groups);
        optimizer.state().clear(); // optimizer holds momentum associated to the old parameters
        optimizer.zero_grad();     // old gradients associated to old parameters might have stale shapes
    }
}
