#pragma once
#include <filesystem>
#include <vector>
#include <string>

namespace utils
{
  inline std::vector<std::filesystem::path> listImages(const std::filesystem::path &root)
  {
    std::vector<std::filesystem::path> out;
    for (auto const &d : std::filesystem::recursive_directory_iterator(root))
      if (d.is_regular_file())
      {
        auto e = d.path().extension().string();
        if (e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp")
          out.push_back(d.path());
      }
    return out;
  }
}
