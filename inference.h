#pragma once
#include <iostream>
#include <string>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "depth_anything.h"
cv::Mat inference(cv::Mat& image,  DepthAnything& depth_model);
