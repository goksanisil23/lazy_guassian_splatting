#include "gausplat.hpp"
#include "gsplat_data.hpp"
#include "ssim.hpp"
#include "typedefs.h"
#include "utils.hpp"

#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace
{
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

void step(LearnableParams         &params,
          const gsplat::Camera    &cam,
          torch::optim::Optimizer &optimizer,
          torch::Tensor           &image,
          torch::Tensor           &gt_image_tensor)
{
    optimizer.zero_grad();
    image.reset();

    image = forwardWithCulling(params, cam).permute({2, 0, 1});

    auto loss = gaussianLoss(image, gt_image_tensor);
    loss.backward();
    optimizer.step();

    c10::cuda::CUDACachingAllocator::emptyCache();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_sparse_0>\n";
        return 1;
    }
    std::string        dir = argv[1];
    gsplat::GsplatData data(dir);

    auto params = createLearningParamsFromGaussians(data.gaussians_, data.device_);

    AdamsParams adams;

    std::vector<torch::optim::OptimizerParamGroup> param_groups;

    param_groups.emplace_back(std::vector<torch::Tensor>{params.pws},
                              std::make_unique<torch::optim::AdamOptions>(adams.pws_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.low_shs},
                              std::make_unique<torch::optim::AdamOptions>(adams.low_shs_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.high_shs},
                              std::make_unique<torch::optim::AdamOptions>(adams.high_shs_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.alphas_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams.alphas_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.scales_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams.scales_raw_lr));
    param_groups.emplace_back(std::vector<torch::Tensor>{params.rots_raw},
                              std::make_unique<torch::optim::AdamOptions>(adams.rots_raw_lr));

    torch::optim::Adam optimizer(param_groups, torch::optim::AdamOptions(/*lr=*/0.F));

    // initialize the index map
    gsplat::GLOBAL_IDX_MAP = makeIdxMap(data.cameras_[0].height, data.cameras_[0].width, data.device_);

    const int64_t num_epochs = 300;
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch)
    {
        auto const shuffled_img_indices = generateShuffledIndices(data.images_.size());

        size_t img_ctr = 0;
        for (size_t img_idx : shuffled_img_indices)
        {
            std::cout << "Epoch " << epoch << ", image " << img_ctr << "/" << shuffled_img_indices.size() << std::endl;

            gsplat::Camera &cam             = data.cameras_[img_idx];
            torch::Tensor   gt_image_tensor = data.images_[img_idx];

            torch::Tensor image;
            step(params, cam, optimizer, image, gt_image_tensor);
            // stepChunked(params, cam, optimizer, image, gt_image_tensor);

            if (img_idx == 0)
            {
                // Show the image by copying it to CPU and converting to uint8_t
                auto image_cpu = image.detach().cpu().permute({1, 2, 0}).clamp(0, 1).mul(255).to(torch::kUInt8);
                std::cout << "rendered image on cpu: " << image_cpu.sizes() << "\n";
                cv::Mat img_mat(image_cpu.size(0), image_cpu.size(1), CV_8UC3, image_cpu.data_ptr());
                cv::cvtColor(img_mat, img_mat, cv::COLOR_RGB2BGR);
                cv::imwrite("rendered_image_epoch_" + std::to_string(epoch) + ".png", img_mat);
            }
            img_ctr++;
        }
    }

    return 0;
}
