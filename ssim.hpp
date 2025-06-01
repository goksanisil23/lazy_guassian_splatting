#pragma once

#include <torch/torch.h>

namespace
{
/* Helpers are modified from
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
*/

// Create a 1D Gaussian kernel of length `window_size` and standard
// deviation `sigma`. Returned tensor is float32 on CPU.
torch::Tensor gaussian(const int window_size, const double sigma)
{
    auto      gauss  = torch::empty({window_size}, torch::kFloat32);
    const int center = window_size / 2;
    for (int i = 0; i < window_size; ++i)
    {
        float val =
            std::exp(-static_cast<float>((i - center) * (i - center)) / static_cast<float>(2.0 * sigma * sigma));
        gauss[i] = val;
    }
    gauss = gauss / gauss.sum();
    return gauss; // shape [window_size], CPU float32
}

// Build a 2D separable window of size [channel, 1, window_size, window_size].
// Under the hood, we compute outer product of the 1D Gaussian with itself.
// Returned tensor is float32 on CPU. Later, in ssim(), we will .to(device).to(dtype).
torch::Tensor createWindow(const int window_size, const int64_t channel)
{
    // 1D kernel [window_size] → column vector [window_size, 1]
    torch::Tensor _1D_window = gaussian(window_size, 1.5).unsqueeze(1); // [W,1]

    // Outer product → 2D kernel [window_size, window_size]
    torch::Tensor _2D = _1D_window.mm(_1D_window.t()).to(torch::kFloat32); // [W,W]

    // Reshape to [1,1,W,W], then expand to [channel,1,W,W]
    torch::Tensor _2D_window = _2D.unsqueeze(0)
                                   .unsqueeze(0) // [1,1,W,W]
                                   .expand({channel, 1, window_size, window_size})
                                   .contiguous(); // [C,1,W,W]

    return _2D_window; // CPU float32
}

// Internal SSIM computation. Expects:
//   - img1, img2: tensors of shape [N, C, H, W], same device & dtype.
//   - window:    tensor of shape [C, 1, window_size, window_size], same
//                device & dtype as img1.
//   - window_size, channel: ints.
//   - size_average: if true, return single scalar; else return tensor [N].
// Returns:
//   - if size_average: scalar SSIM (torch::Tensor with no dims).
//   - else: a 1D tensor of length N, SSIM per batch element.
torch::Tensor _ssim(const torch::Tensor &img1,
                    const torch::Tensor &img2,
                    const torch::Tensor &window,
                    const int            window_size,
                    const int64_t        channel,
                    const bool           size_average)
{
    namespace F = torch::nn::functional;

    // convolution options: same padding = window_size//2, groups = channel
    F::Conv2dFuncOptions conv_opts;
    conv_opts.padding(window_size / 2).groups(channel);

    // μ1 = conv2d(img1, window)
    torch::Tensor mu1 = F::conv2d(img1, window, conv_opts);
    torch::Tensor mu2 = F::conv2d(img2, window, conv_opts);

    torch::Tensor mu1_sq  = mu1.pow(2);
    torch::Tensor mu2_sq  = mu2.pow(2);
    torch::Tensor mu1_mu2 = mu1 * mu2;

    // σ1² = conv2d(img1*img1, window) - μ1²
    torch::Tensor sigma1_sq = F::conv2d(img1 * img1, window, conv_opts) - mu1_sq;
    torch::Tensor sigma2_sq = F::conv2d(img2 * img2, window, conv_opts) - mu2_sq;
    torch::Tensor sigma12   = F::conv2d(img1 * img2, window, conv_opts) - mu1_mu2;

    // Constants per original SSIM paper
    constexpr double C1 = 0.01 * 0.01; // 0.0001
    constexpr double C2 = 0.03 * 0.03; // 0.0009

    // SSIM map: ((2 μ1μ2 + C1)*(2 σ12 + C2)) / ((μ1² + μ2² + C1)*(σ1² + σ2² + C2))
    torch::Tensor numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2);
    torch::Tensor denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);
    torch::Tensor ssim_map    = numerator / denominator; // [N, C, H, W]

    if (size_average)
    {
        return ssim_map.mean(); // scalar
    }
    else
    {
        // mean over channel, height, width → shape [N]
        return ssim_map.mean({1, 2, 3});
    }
}

// Public SSIM function. Builds the Gaussian window, moves it to the same device
// & dtype as img1, then calls _ssim. Expects:
//   - img1, img2: [N, C, H, W], same shape, same device, same dtype.
//   - window_size (default=11), size_average (default=true).
// Returns same as _ssim.
torch::Tensor
ssim(const torch::Tensor &img1, const torch::Tensor &img2, const int window_size = 11, const bool size_average = true)
{
    int64_t channel = img1.size(0); // C dimension

    // Create CPU float32 window [C,1,window_size,window_size]
    torch::Tensor window = createWindow(window_size, channel);

    // Move window to img1's device & dtype
    window = window.to(img1.device()).to(img1.dtype());

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

} // namespace

torch::Tensor gaussianLoss(const torch::Tensor &image, const torch::Tensor &gt_image)
{
    constexpr float kLossLambda = 0.2f;

    // L1 loss: mean absolute difference
    torch::Tensor loss_l1 = torch::abs(image - gt_image).mean();

    // SSIM loss: 1 - SSIM(image, gt_image)
    torch::Tensor loss_ssim = 1.0f - ssim(image, gt_image);

    // Combine losses
    return (1.0f - kLossLambda) * loss_l1 + kLossLambda * loss_ssim;
}