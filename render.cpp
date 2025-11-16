#include "adaptive_densification.hpp"
#include "gausplat.hpp"
#include "gsplat_data.hpp"
#include "io.hpp"
#include "params_to_gaussians.hpp"
#include "plot_helper.hpp"
#include "ssim.hpp"
#include "typedefs.h"
#include "utils.hpp"

#include "raylib-cpp.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace
{

constexpr size_t kImageIdxToShow = 0U;
constexpr bool   kUseHigherShs   = false;

} // namespace

void step(LearnableParams &params, torch::Tensor &image, const gsplat::Camera &cam)
{

    auto          t0 = std::chrono::high_resolution_clock::now();
    torch::Tensor depths_in_cam_frame;
    torch::Tensor gaus_centers_img_frame = torch::zeros(
        {params.pws.sizes()[0], 2}, torch::TensorOptions().device(params.pws.device()).dtype(torch::kFloat32));
    image   = forwardWithCulling(params, cam, depths_in_cam_frame, gaus_centers_img_frame).permute({2, 0, 1});
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "forward: " << (t1 - t0).count() * 1e-6 << " ms" << std::endl;
}

void updateGsplatCamera(gsplat::Camera     &camera,
                        const Vector3      &ray_cam_pos,
                        const Vector3      &forward,
                        const Vector3      &right,
                        const Vector3      &up,
                        const torch::Device device)
{
    camera.tcw = torch::tensor({ray_cam_pos.x, ray_cam_pos.y, ray_cam_pos.z},
                               torch::TensorOptions().dtype(torch::kFloat).device(device));
    camera.Rcw = torch::stack(
        {torch::tensor({right.x, right.y, right.z}, torch::TensorOptions().dtype(torch::kFloat).device(device)),
         torch::tensor({up.x, up.y, up.z}, torch::TensorOptions().dtype(torch::kFloat).device(device)),
         torch::tensor({-forward.x, -forward.y, -forward.z},
                       torch::TensorOptions().dtype(torch::kFloat).device(device))},
        /*dim=*/0);

    camera.twc = -camera.Rcw.transpose(0, 1).matmul(camera.tcw.unsqueeze(1)).squeeze();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_gaussian>\n";
        return 1;
    }
    std::string       dir = argv[1];
    gsplat::Gaussians gaussians;
    bool const        load_ok = loadGaussiansFromFile(gaussians, "gaussians.bin");
    if (!load_ok)
    {
        std::cerr << "Failed to load gaussians from file." << std::endl;
        return 1;
    }
    auto const device = torch::kCUDA;

    auto params = createLearningParamsFromGaussians(gaussians, device, kUseHigherShs);

    // Pick a camera
    gsplat::Camera cam;
    cam.id     = 0;
    cam.width  = 490;
    cam.height = 272;
    cam.fx     = 290.045;
    cam.fy     = 290.631;
    cam.cx     = 245;
    cam.cy     = 136;
    // Initial camera pose
    cam.Rcw = torch::tensor({{0.9871f, -0.0005f, -0.1601f}, {0.0118f, 0.9975f, 0.0695f}, {0.1597f, -0.0704f, 0.9846f}},
                            torch::TensorOptions().dtype(torch::kFloat).device(device));
    cam.tcw = torch::tensor({2.5204, 0.4451, 4.5644}, torch::TensorOptions().dtype(torch::kFloat).device(device));
    cam.twc = -cam.Rcw.transpose(0, 1).matmul(cam.tcw.unsqueeze(1)).squeeze();

    Vector3 cam_pos = {2.5204f, 0.4451f, 4.5644f};

    constexpr int64_t max_height = 700;
    constexpr int64_t max_width  = 500;
    gsplat::GLOBAL_IDX_MAP       = makeIdxMap(max_height, max_width, device);

    constexpr float move_speed = 0.05f;
    constexpr float rot_speed  = 0.03f; // radians per frame

    float yaw   = 0.0f;
    float pitch = 0.0f;

    Texture2D texture = {0};
    bool      is_first{true};

    raylib::Window window(cam.width, cam.height, "GSplat Renderer");
    SetTargetFPS(10);

    while (!window.ShouldClose())
    {

        if (IsKeyDown(KEY_RIGHT))
            yaw -= rot_speed;
        if (IsKeyDown(KEY_LEFT))
            yaw += rot_speed;

        if (IsKeyDown(KEY_UP))
            pitch += rot_speed;
        if (IsKeyDown(KEY_DOWN))
            pitch -= rot_speed;

        // --------- Basis from yaw/pitch ----------
        Vector3 forward = {cosf(pitch) * cosf(yaw), sinf(pitch), cosf(pitch) * sinf(yaw)};
        forward         = Vector3Normalize(forward);

        Vector3 worldUp = {0.0f, 1.0f, 0.0f};
        Vector3 right   = Vector3Normalize(Vector3CrossProduct(forward, worldUp));
        Vector3 up      = Vector3Normalize(Vector3CrossProduct(right, forward));

        // --------- WASD translation in local frame ----------
        if (IsKeyDown(KEY_W))
            cam_pos = Vector3Add(cam_pos, Vector3Scale(forward, move_speed));
        if (IsKeyDown(KEY_S))
            cam_pos = Vector3Subtract(cam_pos, Vector3Scale(forward, move_speed));
        if (IsKeyDown(KEY_D))
            cam_pos = Vector3Add(cam_pos, Vector3Scale(right, move_speed));
        if (IsKeyDown(KEY_A))
            cam_pos = Vector3Subtract(cam_pos, Vector3Scale(right, move_speed));

        updateGsplatCamera(cam, cam_pos, forward, right, up, device);

        torch::Tensor image;
        step(params, image, cam);

        auto image_cpu = image.detach().cpu().permute({1, 2, 0}).clamp(0, 1).mul(255).to(torch::kUInt8);

        Image frame_img{};
        frame_img.data    = image_cpu.data_ptr();
        frame_img.width   = static_cast<int>(cam.width);
        frame_img.height  = static_cast<int>(cam.height);
        frame_img.mipmaps = 1;
        frame_img.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8;

        if (is_first)
        {
            texture  = LoadTextureFromImage(frame_img);
            is_first = false;
        }
        else
        {
            UpdateTexture(texture, frame_img.data);
        }

        window.BeginDrawing();
        window.ClearBackground(BLACK);

        if (!is_first)
            DrawTexture(texture, 0, 0, WHITE);

        DrawFPS(10, 10);
        window.EndDrawing();
    }

    UnloadTexture(texture);

    return 0;
}
