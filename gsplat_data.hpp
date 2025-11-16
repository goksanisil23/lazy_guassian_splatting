#include "colmap_loader.hpp"
#include "spherical_harmonics_coefs.h"
#include "typedefs.h"
#include <algorithm>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <torch/torch.h>

namespace
{

constexpr float kScaleDownFactor = 2.0;

torch::Tensor qVec2RotMat(const std::array<float, 4> &qvec)
{
    const float q0 = qvec[0];
    const float q1 = qvec[1];
    const float q2 = qvec[2];
    const float q3 = qvec[3];

    std::array<float, 9> rot_mtx{{1 - 2 * q2 * q2 - 2 * q3 * q3,
                                  2 * q1 * q2 - 2 * q0 * q3,
                                  2 * q3 * q1 + 2 * q0 * q2,
                                  2 * q1 * q2 + 2 * q0 * q3,
                                  1 - 2 * q1 * q1 - 2 * q3 * q3,
                                  2 * q2 * q3 - 2 * q0 * q1,
                                  2 * q3 * q1 - 2 * q0 * q2,
                                  2 * q2 * q3 + 2 * q0 * q1,
                                  1 - 2 * q1 * q1 - 2 * q2 * q2}};

    auto rot_mtx_tensor =
        torch::from_blob((void *)(rot_mtx.data()), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    return rot_mtx_tensor;
}

cv::Mat loadSingleImage(const std::string &im_path)
{
    cv::Mat img = cv::imread(im_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + im_path);
    }
    // resize by downscale factor
    cv::resize(img, img, cv::Size(), 1.0f / kScaleDownFactor, 1.0f / kScaleDownFactor, cv::INTER_LINEAR);
    return img;
}

torch::Tensor
loadImageTensor(const std::string &im_path, torch::Device device, const int64_t cam_width, const int64_t cam_height)
{
    cv::Mat img = cv::imread(im_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(cam_width, cam_height), 0, 0, cv::INTER_LINEAR);
    // create tensor from H×W×C uint8
    auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8);
    // convert to C×H×W float on device, normalized [0,1]
    tensor = tensor.permute({2, 0, 1}).to(device, torch::kFloat32).div(255.0);
    return tensor;
}

// Estimate the scene scale based on the furthest camera distance from the scene center
float findSceneScale(const std::vector<gsplat::Camera> &cameras)
{
    std::vector<torch::Tensor> twc_list;
    twc_list.reserve(cameras.size());
    for (auto const &cam : cameras)
    {
        twc_list.push_back(cam.twc.clone());
    }
    auto const twcs = torch::stack(twc_list, 0); // [num_cams, 3]
    const auto cam_dist =
        torch::linalg_norm(twcs - torch::mean(twcs, 0), /*ord*/ 2, /*dim=*/c10::IntArrayRef{1}); // [num_cams]
    constexpr float kScaleFactor = 1.1F;
    const float     scene_scale  = torch::max(cam_dist).item<float>() * kScaleFactor;
    std::cout << "scene scale: " << scene_scale << std::endl;
    return scene_scale;
}

} // namespace

namespace gsplat
{

float findNearestDist(const std::vector<Vec3d> &pts, const size_t &query_idx)
{
    float       bestDist = std::numeric_limits<float>::infinity();
    auto const &query_pt = pts[query_idx];
    size_t      best_idx = query_idx;
    for (size_t j = 0; j < pts.size(); ++j)
    {
        if (query_idx == j)
            continue;
        float dx = query_pt.x - pts[j].x;
        float dy = query_pt.y - pts[j].y;
        float dz = query_pt.z - pts[j].z;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestDist)
        {
            best_idx = j;
            bestDist = d2;
        }
    }

    assert(best_idx != query_idx);
    return std::sqrt(bestDist);
}

Gaussians initGaussiansFrom3dPoints(const std::vector<colmap_loader::Point3D> &pts)
{
    constexpr float kInitialAlpha = 0.8f;
    Gaussians       gaussians;

    gaussians.pws.reserve(pts.size());
    gaussians.shs.reserve(pts.size());
    gaussians.scales.reserve(pts.size());
    gaussians.rots.reserve(pts.size());
    gaussians.alphas.reserve(pts.size());

    for (size_t i = 0; i < pts.size(); ++i)
    {
        auto const &p = pts[i];
        gaussians.pws.push_back({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)});
        gaussians.shs.push_back({((static_cast<float>(p.r) / 255.0f) - 0.5f) / SH_C0_0,
                                 ((static_cast<float>(p.g) / 255.0f) - 0.5f) / SH_C0_0,
                                 ((static_cast<float>(p.b) / 255.0f) - 0.5f) / SH_C0_0});

        gaussians.rots.push_back({0, 0, 0, 1});

        gaussians.alphas.push_back(kInitialAlpha);
    }

    for (size_t i = 0; i < pts.size(); ++i)
    {
        // TODO: enable
        // float nearest_dist = findNearestDist(gaussians.pws, i);
        static_cast<void>(findNearestDist);
        float nearest_dist = 0.1f;
        // Clip it to [0,01, 3]
        // nearest_dist = std::max(0.01f, std::min(3.0f, nearest_dist));
        // Use the nearest distance as isomorphic gaussian scale
        gaussians.scales.push_back({nearest_dist, nearest_dist, nearest_dist});
    }

    return gaussians;
}

class GsplatData
{
  public:
    const torch::Device device_{torch::kCUDA};

    GsplatData(const std::string &dataset_path, const uint32_t max_num_gaussians, const uint32_t max_num_images)
    {
        auto cams = colmap_loader::loadCameras(dataset_path + "/sparse/0/cameras.bin");
        auto ims  = colmap_loader::loadImages(dataset_path + "/sparse/0/images.bin");
        auto pts  = colmap_loader::loadPoints(dataset_path + "/sparse/0/points3D.bin");

        std::cout << "Loaded: " << cams.size() << " cameras, " << ims.size() << " images, " << pts.size() << " points3D"
                  << std::endl;
        colmap_loader::summarize(cams, ims, pts);

        for (const colmap_loader::Image &im : ims)
        {
            auto const  actual_img = loadSingleImage(dataset_path + "/images/" + im.name);
            const float w_scale    = actual_img.cols / static_cast<float>(cams[im.camera_id].w);
            const float h_scale    = actual_img.rows / static_cast<float>(cams[im.camera_id].h);

            Camera cam;
            cam.id     = im.id;
            cam.width  = static_cast<int64_t>(static_cast<float>(cams[im.camera_id].w) * w_scale);
            cam.height = static_cast<int64_t>(static_cast<float>(cams[im.camera_id].h) * h_scale);
            cam.fx     = cams[im.camera_id].params[0] * w_scale;
            cam.fy     = cams[im.camera_id].params[1] * h_scale;
            cam.cx     = cams[im.camera_id].params[2] * w_scale;
            cam.cy     = cams[im.camera_id].params[3] * h_scale;

            cam.Rcw = qVec2RotMat(im.q).to(device_, torch::kFloat32);
            cam.tcw = torch::from_blob((void *)(im.t.data()), {3}, torch::kFloat32).to(device_, torch::kFloat32);
            cam.twc = -cam.Rcw.transpose(0, 1).matmul(cam.tcw.unsqueeze(1)).squeeze();

            cam.image_path          = dataset_path + "/images/" + im.name;
            auto const image_tensor = loadImageTensor(cam.image_path, device_, cam.width, cam.height);

            images_.push_back(image_tensor);
            cameras_.push_back(cam);
        }

        gaussians_ = initGaussiansFrom3dPoints(pts);

        // Limit the data
        {
            std::random_device rd;
            std::mt19937       gen(rd());

            std::shuffle(gaussians_.pws.begin(), gaussians_.pws.end(), gen);
            std::shuffle(gaussians_.shs.begin(), gaussians_.shs.end(), gen);
            std::shuffle(gaussians_.scales.begin(), gaussians_.scales.end(), gen);
            std::shuffle(gaussians_.rots.begin(), gaussians_.rots.end(), gen);
            std::shuffle(gaussians_.alphas.begin(), gaussians_.alphas.end(), gen);

            // std::shuffle(images_.begin(), images_.end(), gen);
            // std::shuffle(cameras_.begin(), cameras_.end(), gen);
            std::reverse(images_.begin(), images_.end());
            std::reverse(cameras_.begin(), cameras_.end());

            gaussians_.pws.resize(max_num_gaussians);
            gaussians_.shs.resize(max_num_gaussians);
            gaussians_.scales.resize(max_num_gaussians);
            gaussians_.rots.resize(max_num_gaussians);
            gaussians_.alphas.resize(max_num_gaussians);
            images_.resize(max_num_images);
        }

        scene_scale_ = findSceneScale(cameras_);
    }

  public:
    std::vector<Camera>        cameras_;
    std::vector<torch::Tensor> images_;
    Gaussians                  gaussians_;

    float scene_scale_{0.F};
};
} // namespace gsplat