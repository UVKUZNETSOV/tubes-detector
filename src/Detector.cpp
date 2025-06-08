#include "../include/Detector.h"
using cv::Mat;
using cv::Vec3f;

TubeDetector::TubeDetector(const TubeParams &p) : params_(p) {}

static inline double dist2(const Vec3f &a, const Vec3f &b)
{
  double dx = a[0] - b[0], dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

void TubeDetector::merge(std::vector<Vec3f> &dst,
                         const std::vector<Vec3f> &add,
                         int tol)
{
  double tol2 = tol * tol;
  for (auto &c : add)
  {
    bool dup = false;
    for (auto &d : dst)
      if (dist2(c, d) < tol2)
      {
        dup = true;
        break;
      }
    if (!dup)
      dst.push_back(c);
  }
}

std::vector<Vec3f> TubeDetector::detectByContours(const Mat &gray,
                                                  const TubeParams &p)
{
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  clahe->setClipLimit(p.clahe_clip);
  Mat enh;
  clahe->apply(gray, enh);

  Mat bin;
  cv::adaptiveThreshold(enh, bin, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV, 11, 2);
  cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}), {-1, -1}, 2);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<Vec3f> circles;
  for (auto &c : contours)
  {
    double area = cv::contourArea(c);
    if (area < p.minArea || area > p.maxArea)
      continue;
    double peri = cv::arcLength(c, true);
    double circ = (peri == 0) ? 0 : 4 * CV_PI * area / (peri * peri);
    if (circ < p.circularity)
      continue;

    cv::Point2f center;
    float r;
    cv::minEnclosingCircle(c, center, r);
    if (r < p.minRadius || r > p.maxRadius)
      continue;
    circles.emplace_back(center.x, center.y, r);
  }
  return circles;
}

std::vector<Vec3f> TubeDetector::detect(const Mat &bgr) const
{
  Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  if (params_.blur > 1)
    cv::medianBlur(gray, gray, params_.blur);

 
  std::vector<Vec3f> hough;
  cv::HoughCircles(gray, hough, cv::HOUGH_GRADIENT,
                   1, gray.rows / 8,
                   params_.canny,
                   params_.accumulator,
                   params_.minRadius,
                   params_.maxRadius);

 
  std::vector<Vec3f> cont = detectByContours(gray, params_);

 
  merge(hough, cont, params_.duplicateThresh);

  return hough;
}
