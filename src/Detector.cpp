#include "../include/Detector.h"
#include <numeric>
#include <algorithm>

using cv::Mat;
using cv::Vec3f;

static inline double d2(const Vec3f &a, const Vec3f &b)
{
  double dx = a[0] - b[0], dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

void TubeDetector::merge(std::vector<Vec3f> &dst,
                         const std::vector<Vec3f> &add,
                         int tol)
{
  double t2 = tol * tol;
  for (const auto &c : add)
  {
    bool dup = false;
    for (auto &d : dst)
      if (d2(c, d) < t2)
      {
        dup = true;
        break;
      }
    if (!dup)
      dst.push_back(c);
  }
}

static std::vector<int>
findPeaks(const std::vector<int> &v, int want, int minDist)
{
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b)
            { return v[a] > v[b]; });

  std::vector<int> peaks;
  for (int i : idx)
  {
    bool far = true;
    for (int p : peaks)
      if (std::abs(i - p) < minDist)
      {
        far = false;
        break;
      }
    if (far)
    {
      peaks.push_back(i);
      if ((int)peaks.size() == want)
        break;
    }
  }
  std::sort(peaks.begin(), peaks.end());
  return peaks;
}

static std::vector<int>
collapse(const std::vector<int> &src, int mergePX, int want)
{
  std::vector<int> out;
  for (int p : src)
  {
    if (out.empty() || p - out.back() > mergePX)
      out.push_back(p);
    else
      out.back() = (out.back() + p) / 2;
    if ((int)out.size() == want)
      break;
  }
  return out;
}

std::vector<Vec3f> TubeDetector::detectGrid(const Mat &gray) const
{
  Mat inv = 255 - gray;

  Mat projY;
  cv::reduce(inv, projY, 1, cv::REDUCE_SUM, CV_32S);
  std::vector<int> vY(projY.rows);
  for (int y = 0; y < projY.rows; ++y)
    vY[y] = projY.at<int>(y);

  auto rowPeaks = findPeaks(vY, params_.rows, params_.bandH);
  if ((int)rowPeaks.size() != params_.rows)
    return {};

  int band = params_.bandH / 2;
  std::vector<Vec3f> circles;

  for (int cy : rowPeaks)
  {

    int offs = band / 6;
    int y0 = std::max(0, cy - offs);
    int y1 = std::min(gray.rows - 1, cy + offs);
    Mat bandMat = inv.rowRange(y0, y1 + 1);

    Mat projX;
    cv::reduce(bandMat, projX, 0, cv::REDUCE_SUM, CV_32S);
    std::vector<int> vX(projX.cols);
    for (int x = 0; x < projX.cols; ++x)
      vX[x] = projX.at<int>(x);

    int maxVal = *std::max_element(vX.begin(), vX.end());
    int thrVal = static_cast<int>(maxVal * 0.3);
    for (int &v : vX)
      if (v < thrVal)
        v = 0;

    auto raw = findPeaks(vX, params_.cols * 4, 5);
    auto cols = collapse(raw, params_.peakTol, params_.cols);
    if ((int)cols.size() != params_.cols)
      return {};

    double pitch = double(cols.back() - cols.front()) / (params_.cols - 1);
    double rGoodMin = 0.35 * pitch;
    double rGoodMax = 0.55 * pitch;
    double rNominal = 0.45 * pitch;

    for (int cx : cols)
    {
      if (rNominal < rGoodMin || rNominal > rGoodMax)
        continue;
      circles.emplace_back(cx, cy, rNominal);
    }
  }
  return circles;
}

std::vector<Vec3f> TubeDetector::detectHough(const Mat &gray) const
{
  std::vector<Vec3f> res, tmp;
  cv::HoughCircles(gray, tmp, cv::HOUGH_GRADIENT, 1, params_.minDist,
                   params_.canny, params_.accOuter,
                   params_.minROuter, params_.maxROuter);
  merge(res, tmp, params_.mergeTol);

  tmp.clear();
  cv::HoughCircles(gray, tmp, cv::HOUGH_GRADIENT, 1, params_.minDist,
                   params_.canny, params_.accInner,
                   params_.minRInner, params_.maxRInner);
  merge(res, tmp, params_.mergeTol);
  return res;
}

TubeDetector::TubeDetector(const TubeParams &p) : params_(p) {}

std::vector<Vec3f> TubeDetector::detect(const Mat &bgr) const
{
  Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  if (params_.blur > 1)
    cv::medianBlur(gray, gray, params_.blur);

  if (params_.useGrid)
  {
    auto g = detectGrid(gray);
    if (!g.empty())
      return g;
  }
  return detectHough(gray);
}
