#include "adaptive_densification.hpp"
#include "gausplat.hpp"
#include "gsplat_data.hpp"
#include "plot_helper.hpp"
#include "ssim.hpp"
#include "typedefs.h"
#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace
{

constexpr bool    kEnableAdaptiveDensification = true;
constexpr int64_t kNumGaussiansToLoad          = 10000;
constexpr int64_t kNumGaussiansLimit           = 25000; // set based on available GPU memory
constexpr int64_t kNumEpochs                   = 300;
constexpr size_t  kImageIdxToShow              = 0U;

std::vector<size_t> generateShuffledIndices(const size_t N)
{
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);                                       // Fill with 0, 1, ..., N-1
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()}); // Shuffle
    return indices;
}

} // namespace

LearnableParams createLearningParamsFromGaussians(const gsplat::Gaussians &gaussians, const torch::Device &device)
{
    LearnableParams params;
    // 1) pws
    params.pws = torch::from_blob((void *)(gaussians.pws.data()),
                                  {(long)gaussians.pws.size(), 3},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .to(device)
                     .set_requires_grad(true); // [N,3]

    // 2) rots_raw
    params.rots_raw = torch::from_blob((void *)(gaussians.rots.data()),
                                       {(long)gaussians.rots.size(), 4},
                                       torch::TensorOptions().dtype(torch::kFloat32))
                          .to(device)
                          .set_requires_grad(true); // [N,4]

    // 3) scales_raw
    params.scales_raw = getScalesRaw(torch::from_blob((void *)(gaussians.scales.data()),
                                                      {(long)gaussians.scales.size(), 3},
                                                      torch::TensorOptions().dtype(torch::kFloat32))
                                         .to(device))
                            .set_requires_grad(true); // [N,3]

    // 4) alphas_raw
    params.alphas_raw = getAlphasRaw(torch::from_blob((void *)(gaussians.alphas.data()),
                                                      {(long)gaussians.alphas.size(), 1},
                                                      torch::TensorOptions().dtype(torch::kFloat32))
                                         .to(device))
                            .set_requires_grad(true); // [N,1]

    // 5) low_shs & high_shs
    params.low_shs = torch::from_blob((void *)(gaussians.shs.data()),
                                      {(long)gaussians.shs.size(), 3},
                                      torch::TensorOptions().dtype(torch::kFloat32))
                         .to(device)
                         .set_requires_grad(true); // [N,3]
    params.high_shs =
        torch::ones_like(params.low_shs).repeat({1, 15}).mul(0.001f).to(device).set_requires_grad(true); // [N,15]

    return params;
}

float step(LearnableParams         &params,
           const gsplat::Camera    &cam,
           torch::optim::Optimizer &optimizer,
           GradAccumInfo           &grad_accum_info,
           torch::Tensor           &image,
           torch::Tensor           &gt_image_tensor)
{
    optimizer.zero_grad();
    image.reset();

    constexpr float kGradAccumDepthThresh = 0.2F;

    auto          t0 = std::chrono::high_resolution_clock::now();
    torch::Tensor depths_in_cam_frame;
    torch::Tensor gaus_centers_img_frame = torch::zeros(
        {params.pws.sizes()[0], 2}, torch::TensorOptions().device(params.pws.device()).dtype(torch::kFloat32));
    image = forwardWithCulling(params, cam, depths_in_cam_frame, gaus_centers_img_frame).permute({2, 0, 1});
    auto const grad_accum_mask = depths_in_cam_frame.gt(kGradAccumDepthThresh);
    auto       t1              = std::chrono::high_resolution_clock::now();
    auto       loss            = gaussianLoss(image, gt_image_tensor);
    loss.backward();
    updateAccumulatedGradInfo(grad_accum_info, gaus_centers_img_frame, grad_accum_mask);
    optimizer.step();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "forward: " << (t1 - t0).count() * 1e-6 << " ms, " << "backward: " << (t2 - t1).count() * 1e-6 << " ms"
              << std::endl;

    return loss.item<float>();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_sparse_0>\n";
        return 1;
    }
    std::string        dir                   = argv[1];
    const int64_t      num_gaussians_to_load = kEnableAdaptiveDensification ? kNumGaussiansToLoad : kNumGaussiansLimit;
    gsplat::GsplatData data(dir, num_gaussians_to_load);

    auto params = createLearningParamsFromGaussians(data.gaussians_, data.device_);
    std::vector<torch::optim::OptimizerParamGroup> param_groups{createAdamParamGroup(params)};
    torch::optim::Adam                             optimizer(param_groups);

    // initialize the index map
    // Find the max camera height and width, so that we have enough indices for all cameras
    int64_t max_height = 0;
    int64_t max_width  = 0;
    for (const auto &cam : data.cameras_)
    {
        if (cam.height > max_height)
            max_height = cam.height;
        if (cam.width > max_width)
            max_width = cam.width;
    }
    gsplat::GLOBAL_IDX_MAP = makeIdxMap(max_height, max_width, data.device_);

    GradAccumInfo grad_accum_info;

    std::vector<float> avg_losses;
    cv::Mat            loss_plot;

    for (int64_t epoch = 0; epoch < kNumEpochs; ++epoch)
    {
        auto const shuffled_img_indices = generateShuffledIndices(data.images_.size());

        size_t img_ctr  = 0;
        float  avg_loss = 0.F;
        for (size_t img_idx : shuffled_img_indices)
        {
            std::cout << "Epoch " << epoch << ", image " << img_ctr << "/" << shuffled_img_indices.size() << std::endl;

            gsplat::Camera &cam             = data.cameras_[img_idx];
            torch::Tensor   gt_image_tensor = data.images_[img_idx];

            torch::Tensor image;
            const float   loss = step(params, cam, optimizer, grad_accum_info, image, gt_image_tensor);

            if (img_idx == kImageIdxToShow)
            {
                // Show the image by copying it to CPU and converting to uint8_t
                auto    image_cpu = image.detach().cpu().permute({1, 2, 0}).clamp(0, 1).mul(255).to(torch::kUInt8);
                cv::Mat img_mat(image_cpu.size(0), image_cpu.size(1), CV_8UC3, image_cpu.data_ptr());
                cv::cvtColor(img_mat, img_mat, cv::COLOR_RGB2BGR);
                cv::imwrite("rendered_image_epoch_" + std::to_string(epoch) + ".png", img_mat);
                cv::imshow("Rendered Image", img_mat);
                cv::waitKey(1);
            }
            img_ctr++;
            avg_loss += loss;
        }
        avg_loss /= static_cast<float>(shuffled_img_indices.size());
        avg_losses.push_back(avg_loss);
        loss_plot = plotLosses(avg_losses);

        if (kEnableAdaptiveDensification && (epoch % 10 == 0) && (epoch > 0))
        {
            if (params.pws.size(0) >= kNumGaussiansLimit)
            {
                std::cout << "skipping densification since number of gaussians reached limit: " << params.pws.size(0)
                          << std::endl;
            }
            else
            {
                adaptiveDensification(params, optimizer, grad_accum_info, data.scene_scale_);
                std::cout << "Gaussians after densification: " << params.pws.sizes() << std::endl;
                // Reset the grad accumulation info
                grad_accum_info.reset();
            }
        }
    }
    if (!loss_plot.empty())
    {
        cv::imwrite("training_loss.png", loss_plot);
    }

    return 0;
}
