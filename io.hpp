#pragma once

#include <fstream>
#include <vector>

#include "typedefs.h"

template <typename T>
bool writePOD(std::ostream &os, const T &value)
{
    return os.write(reinterpret_cast<const char *>(&value), sizeof(T)).good();
}

template <typename T>
bool readPOD(std::istream &is, T &value)
{
    return is.read(reinterpret_cast<char *>(&value), sizeof(T)).good();
}

template <typename T>
bool writeVector(std::ostream &os, const std::vector<T> &vec)
{
    const uint64_t size = static_cast<uint64_t>(vec.size());
    if (!writePOD(os, size))
        return false;
    if (size > 0)
    {
        return os.write(reinterpret_cast<const char *>(vec.data()), sizeof(T) * size).good();
    }
    return true;
}

template <typename T>
bool readVector(std::istream &is, std::vector<T> &vec)
{
    uint64_t size = 0;
    if (!readPOD(is, size))
        return false;
    vec.resize(size);
    if (size > 0)
    {
        return is.read(reinterpret_cast<char *>(vec.data()), sizeof(T) * size).good();
    }
    return true;
}

bool saveGaussiansToFile(const std::string &file_path, const gsplat::Gaussians &gaussians)
{
    std::ofstream os(file_path, std::ios::binary);
    if (!os)
        return false;

    if (!writeVector(os, gaussians.pws))
        return false;
    if (!writeVector(os, gaussians.shs))
        return false;
    if (!writeVector(os, gaussians.scales))
        return false;
    if (!writeVector(os, gaussians.rots))
        return false;
    if (!writeVector(os, gaussians.alphas))
        return false;

    return true;
}

bool loadGaussiansFromFile(gsplat::Gaussians &gaussians, const std::string &path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        return false;

    if (!readVector(is, gaussians.pws))
        return false;
    if (!readVector(is, gaussians.shs))
        return false;
    if (!readVector(is, gaussians.scales))
        return false;
    if (!readVector(is, gaussians.rots))
        return false;
    if (!readVector(is, gaussians.alphas))
        return false;

    // Summarize
    std::cout << "Loaded Gaussians from " << path << " :" << std::endl;
    std::cout << "  Number of Gaussians: " << gaussians.pws.size() << std::endl;
    std::cout << "  Number of SHs: " << gaussians.shs.size() << std::endl;
    std::cout << "  Number of Scales: " << gaussians.scales.size() << std::endl;
    std::cout << "  Number of Rots: " << gaussians.rots.size() << std::endl;
    std::cout << "  Number of Alphas: " << gaussians.alphas.size() << std::endl;

    return true;
}