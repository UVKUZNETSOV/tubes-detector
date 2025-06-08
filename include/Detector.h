
#pragma once
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

struct TubeParams
{
  int canny = 100;
  int accumulator = 40;
  int minRadius = 15;
  int maxRadius = 60;

  int blur = 5;
  double clahe_clip = 2.0;
  int minArea = 200;
  int maxArea = 10000;
  double circularity = 0.65;
  int duplicateThresh = 10;
};

class TubeDetector
{
public:
  explicit TubeDetector(const TubeParams &p = {});
  std::vector<cv::Vec3f> detect(const cv::Mat &bgr) const;

private:
  TubeParams params_;
  static std::vector<cv::Vec3f> detectByContours(const cv::Mat &, const TubeParams &);
  static void merge(std::vector<cv::Vec3f> &dst,
                    const std::vector<cv::Vec3f> &add,
                    int tol);
};
