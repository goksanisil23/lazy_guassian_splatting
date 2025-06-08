#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace colmap_loader
{

struct Camera
{
    int32_t             id, model_id;
    uint64_t            w, h;
    std::vector<double> params;
};

struct Image
{
    int32_t               id, camera_id;
    std::array<double, 4> q; // quaternion (w, x, y, z)
    std::array<double, 3> t; // translation (tx, ty, tz)
    std::string           name;
};

struct Point3D
{
    int64_t id;
    double  x, y, z;
    uint8_t r, g, b;
};

void dump3dPoints(const std::vector<Point3D> &pts)
{
    //   Write the 3d points as xyz with rgb colors
    std::ofstream out("points.xyz");
    if (!out)
    {
        std::cerr << "Failed to open points.xyz for writing\n";
        return;
    }
    // Optional header for some viewers (skip if not needed):
    out << "# x y z r g b\n";

    for (const auto &p : pts)
    {
        out << p.x << ' ' << p.y << ' ' << p.z << ' ' << int(p.r) << ' ' << int(p.g) << ' ' << int(p.b) << '\n';
    }
}

void summarize(const std::unordered_map<uint32_t, Camera> &cams,
               const std::vector<Image>                   &ims,
               const std::vector<Point3D>                 &pts)
{
    // dump first of each
    // for (auto const &c : cams)
    {
        const auto &c = cams.begin()->second;
        std::cout << "Camera " << c.id << " model_id=" << c.model_id << " size=" << c.w << "x" << c.h
                  << " params[0..] = ";
        for (int i = 0; i < std::min((int)c.params.size(), 4); ++i)
            std::cout << c.params[i] << " ";
        std::cout << "\n";
    }
    // for (auto const &im : ims)
    {
        const auto &im = ims[0];
        std::cout << "Image " << im.id << " name=" << im.name << " cam=" << im.camera_id << " t=[" << im.t[0] << ","
                  << im.t[1] << "," << im.t[2] << "] q=[" << im.q[0] << "," << im.q[1] << "," << im.q[2] << ","
                  << im.q[3] << "]\n";
    }
    // for (auto const &p : pts)
    {
        const auto &p = pts[0];
        std::cout << "Point3D " << p.id << " xyz=(" << p.x << "," << p.y << "," << p.z << ")" << " rgb=(" << int(p.r)
                  << "," << int(p.g) << "," << int(p.b) << ")\n";
    }
}

// basic readers
uint32_t read_u32(std::ifstream &f)
{
    uint32_t x;
    f.read((char *)&x, sizeof(x));
    return x;
}
int32_t read_i32(std::ifstream &f)
{
    int32_t x;
    f.read((char *)&x, sizeof(x));
    return x;
}
int64_t read_i64(std::ifstream &f)
{
    int64_t x;
    f.read((char *)&x, sizeof(x));
    return x;
}
uint64_t read_u64(std::ifstream &f)
{
    uint64_t x;
    f.read((char *)&x, sizeof(x));
    return x;
}
double read_d(std::ifstream &f)
{
    double x;
    f.read((char *)&x, sizeof(x));
    return x;
}
float read_f(std::ifstream &f)
{
    float x;
    f.read((char *)&x, sizeof(x));
    return x;
}
uint8_t read_u8(std::ifstream &f)
{
    uint8_t x;
    f.read((char *)&x, sizeof(x));
    return x;
}
std::string read_string(std::ifstream &f)
{
    std::string s;
    char        c;
    // read until '\0'
    while (f.get(c) && c != '\0')
    {
        s.push_back(c);
    }
    return s;
}

// how many intrinsics for each model_id (COLMAP enum)
int numParams(int model_id)
{
    switch (model_id)
    {
    case 0:
        return 3; // SIMPLE_PINHOLE
    case 1:
        return 4; // PINHOLE
    case 2:
        return 4; // SIMPLE_RADIAL
    case 3:
        return 5; // RADIAL
    case 4:
        return 8; // OPENCV
    case 5:
        return 12; // OPENCV_FISHEYE
    case 6:
        return 12; // FULL_OPENCV
    case 7:
        return 5; // FOV
    default:
        throw std::runtime_error("Unknown camera model");
    }
}

std::unordered_map<uint32_t, Camera> loadCameras(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path);
    auto                                 N = read_u64(f);
    std::unordered_map<uint32_t, Camera> cams;
    for (auto i = 0U; i < N; ++i)
    {
        Camera c;
        c.id       = read_u32(f);
        c.model_id = read_u32(f);
        c.w        = read_u64(f);
        c.h        = read_u64(f);
        int np     = numParams(c.model_id);
        c.params.resize(np);
        for (int j = 0; j < np; ++j)
            c.params[j] = read_d(f);
        cams[c.id] = c;
    }
    return cams;
}

std::vector<Image> loadImages(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path);
    int64_t            N = read_i64(f);
    std::vector<Image> ims;
    ims.reserve(N);
    for (int64_t i = 0; i < N; ++i)
    {
        Image im;
        im.id        = read_i32(f);
        im.q[0]      = read_d(f); // w
        im.q[1]      = read_d(f); // x
        im.q[2]      = read_d(f); // y
        im.q[3]      = read_d(f); // z
        im.t[0]      = read_d(f);
        im.t[1]      = read_d(f);
        im.t[2]      = read_d(f);
        im.camera_id = read_i32(f);
        im.name      = read_string(f);
        int64_t M    = read_i64(f); // num 2D points
        for (int64_t j = 0; j < M; ++j)
        { // skip them
            read_d(f);
            read_d(f);
            read_i64(f);
        }
        ims.push_back(im);
    }
    return ims;
}

std::vector<Point3D> loadPoints(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open " + path);
    int64_t              N = read_i64(f);
    std::vector<Point3D> pts;
    pts.reserve(N);
    for (int64_t i = 0; i < N; ++i)
    {
        Point3D p;
        p.id = read_i64(f);
        p.x  = read_d(f);
        p.y  = read_d(f);
        p.z  = read_d(f);
        p.r  = read_u8(f);
        p.g  = read_u8(f);
        p.b  = read_u8(f);
        read_d(f);                        // reprojection error
        uint64_t track_len = read_i64(f); // track length
        for (uint64_t j = 0; j < track_len; ++j)
        { // skip track
            read_i32(f);
            read_i32(f);
        }
        pts.push_back(p);
    }
    return pts;
}
} // namespace colmap_loader