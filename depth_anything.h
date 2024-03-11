#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "utils.h"

class DepthAnything
{
public:
	DepthAnything(std::string model_path, nvinfer1::ILogger& logger);
	void show();
	cv::Mat predict(cv::Mat& image);
	~DepthAnything();
	
private:
	static float input_w;
	static float input_h;
	static float mean[3];
	static float std[3];
	static int num_classes;

	std::vector<int> offset;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	void* buffer[2];
	cudaStream_t stream;

	std::vector<float> preprocess(cv::Mat& image);
	std::vector<DepthEstimation> postprocess(std::vector<int> mask, int img_w, int img_h);
	void build(std::string onnxPath, nvinfer1::ILogger& logger);
	bool saveEngine(const std::string& filename);
};
