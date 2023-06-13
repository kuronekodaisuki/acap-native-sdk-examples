#pragma once

#ifdef WIN32
#pragma warning(disable:4819)
#endif // DEBUG

#include <opencv2/core.hpp>


bool RotateAndCrop(cv::Mat src, cv::Mat dest, float cx, float cy, double degree);
cv::Mat RotateAndCrop(cv::Mat src, float cx, float cy, double degree, int width, int height);