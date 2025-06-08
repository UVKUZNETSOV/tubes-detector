#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "../include/Detector.h"
#include "../include/utils.h"

using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;

struct CmdOpt
{
  fs::path input = "images";
  fs::path output = "annotated";
  fs::path jsonFile = "labels.json";
  fs::path config = "";
};

static CmdOpt parse(int argc, char **argv)
{
  CmdOpt o;
  for (int i = 1; i < argc; ++i)
  {
    string s = argv[i];
    if (s == "--input" && i + 1 < argc)
      o.input = argv[++i];
    else if (s == "--output" && i + 1 < argc)
      o.output = argv[++i];
    else if (s == "--json" && i + 1 < argc)
      o.jsonFile = argv[++i];
    else if (s == "--config" && i + 1 < argc)
      o.config = argv[++i];
  }
  return o;
}

int main(int argc, char **argv)
{
  CmdOpt opt = parse(argc, argv);

  TubeParams tp;
  // --- чтение YAML-конфига ---
  if (!opt.config.empty())
  {
    try
    {
      cv::FileStorage fs(opt.config.string(), cv::FileStorage::READ);
      if (fs.isOpened())
      {
        fs["blur"] >> tp.blur;
        fs["canny"] >> tp.canny;

        fs["accOuter"] >> tp.accOuter;
        fs["minROuter"] >> tp.minROuter;
        fs["maxROuter"] >> tp.maxROuter;

        fs["accInner"] >> tp.accInner;
        fs["minRInner"] >> tp.minRInner;
        fs["maxRInner"] >> tp.maxRInner;

        fs["minDist"] >> tp.minDist;
        fs["mergeTol"] >> tp.mergeTol;
      }
      else
      {
        std::cerr << "[WARN] Cannot open config file, using defaults\n";
      }
    }
    catch (const cv::Exception &e)
    {
      std::cerr << "[WARN] Config ignored: " << e.what() << '\n';
    }
  }

  TubeDetector detector(tp);
  fs::create_directories(opt.output);

  ofstream log("detect.log");
  json labels = json::array();

  for (auto &imgPath : utils::listImages(opt.input))
  {
    cv::Mat img = cv::imread(imgPath.string());
    if (img.empty())
    {
      log << imgPath << " 0 Fail\n";
      continue;
    }

    auto circles = detector.detect(img);
    if (circles.empty())
      log << imgPath << " 0 Empty\n";
    else
    {
      log << imgPath << " " << circles.size() << " OK\n";
      for (auto &c : circles)
      {
        cv::Point center{cvRound(c[0]), cvRound(c[1])};
        int r = cvRound(c[2]);
        cv::circle(img, center, r, {0, 255, 0}, 2);
        labels.push_back({{"file", imgPath.string()},
                          {"cx", center.x},
                          {"cy", center.y},
                          {"r", r}});
      }
    }
    fs::path rel = fs::relative(imgPath, opt.input);
    fs::create_directories(opt.output / rel.parent_path());
    cv::imwrite((opt.output / rel).string(), img);
  }
  std::ofstream(opt.jsonFile) << labels.dump(2);
  return 0;
}
