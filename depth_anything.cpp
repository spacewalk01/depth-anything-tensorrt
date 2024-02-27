#include "depth_anything.h"

// set network params
float DepthAnything::input_h = 518;
float DepthAnything::input_w = 518;
int DepthAnything::num_classes = 3;
float DepthAnything::mean[3] = { 123.675, 116.28, 103.53 };
float DepthAnything::std[3] = { 58.395, 57.12, 57.375 };

/**
 * @brief DepthAnything`s constructor
 * @param model_path DepthAnything engine file path
 * @param logger Nvinfer ILogger
*/
DepthAnything::DepthAnything(std::string model_path, nvinfer1::ILogger& logger)
{
    // read the engine file
    std::ifstream engineStream(model_path, std::ios::binary);
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // create tensorrt model
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Define input dimensions
    context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, input_h, input_w));

    // create CUDA stream
    cudaStreamCreate(&stream);
}

/**
 * @brief RTMSeg`s destructor
*/
DepthAnything::~DepthAnything()
{
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
}


/**
 * @brief Display network input and output parameters
*/
void DepthAnything::show()
{
    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        std::cout << "node: " << engine->getBindingName(i) << ", ";
        if (engine->bindingIsInput(i))
        {
            std::cout << "type: input" << ", ";
        }
        else
        {
            std::cout << "type: output" << ", ";
        }
        nvinfer1::Dims dim = engine->getBindingDimensions(i);
        std::cout << "dimensions: ";
        for (int d = 0; d < dim.nbDims; d++)
        {
            std::cout << dim.d[d] << " ";
        }
        std::cout << "\n";
    }
}


/**
 * @brief Network preprocessing function
 * @param image Input image
 * @return Processed Tensor
*/
std::vector<float> DepthAnything::preprocess(cv::Mat& image)
{
    std::tuple<cv::Mat, int, int> resized = resize_depth(image, input_w, input_h);
    cv::Mat resized_image = std::get<0>(resized);
    std::vector<float> input_tensor;
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < resized_image.rows; i++)
        {
            for (int j = 0; j < resized_image.cols; j++)
            {
                input_tensor.emplace_back(((float)resized_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
            }
        }
    }
    return input_tensor;
}

cv::Mat DepthAnything::predict(cv::Mat& image)
{
    cv::Mat clone_image;
    image.copyTo(clone_image);

    int img_w = image.cols;
    int img_h = image.rows;

    // Preprocessing
    std::vector<float> input = preprocess(clone_image);
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], input_h * input_w * sizeof(int));
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Inference depth model
    context->enqueueV2(buffer, stream, nullptr);
    cudaStreamSynchronize(stream);

    // Postprocessing
    std::vector<int> depth_data(input_h * input_w);
    cudaMemcpyAsync(depth_data.data(), buffer[1], input_h * input_w * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(buffer[0]);
    cudaFree(buffer[1]);

    // Convert the entire depth_data vector to a CV_32FC1 Mat
    cv::Mat mask_mat(input_h, input_w, CV_32FC1);
    for (int i = 0; i < input_h; i++) {
        for (int j = 0; j < input_w; j++) {
            // Convert integer value to float
            float float_value = *reinterpret_cast<float*>(&depth_data[i * input_w + j]);

            // Assign the float value to the corresponding pixel in the Mat
            mask_mat.at<float>(i, j) = float_value;
        }
    }
    cv::normalize(mask_mat, mask_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Create a colormap from the depth data
    cv::Mat colormap;
    cv::applyColorMap(mask_mat, colormap, cv::COLORMAP_RAINBOW);

    // Rescale the colormap
    int limX, limY;
    if (img_w > img_h)
    {
        limX = input_w;
        limY = input_w * img_h / img_w;
    }
    else
    {
        limX = input_w * img_w / img_h;
        limY = input_w;
    }
    cv::resize(colormap, colormap, cv::Size(img_w, img_h));

    return colormap;
}
