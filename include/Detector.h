
#pragma once
#include <opencv2/opencv.hpp>

struct TubeParams
{

  int blur = 5;
  int canny = 100;

  int accOuter = 20, minROuter = 60, maxROuter = 130;
  int accInner = 15, minRInner = 15, maxRInner = 50;
  int minDist = 100;
  int mergeTol = 50;

  bool useGrid = true;
  int rows = 2;
  int cols = 14;
  int bandH = 70;
  int peakTol = 60;
};

class TubeDetector
{
public:
  explicit TubeDetector(const TubeParams &p = {});
  std::vector<cv::Vec3f> detect(const cv::Mat &bgr) const;

private:
  TubeParams params_;

  static void merge(std::vector<cv::Vec3f> &dst,
                    const std::vector<cv::Vec3f> &add,
                    int tol);

  std::vector<cv::Vec3f> detectGrid(const cv::Mat &gray) const;
  std::vector<cv::Vec3f> detectHough(const cv::Mat &gray) const;
};
