#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

class TRTModule
{

public:

    TRTModule(string modelPath);

    Mat predict(Mat& inputImage);

    ~TRTModule();

private:

    Mat resizeImage(Mat& img, int inputWidth, int inputHeight);

    void build(string onnxPath, bool isFP16 = false);

    bool saveEngine(const std::string& fileName);

    void deserializeEngine(string enginePath);

    void initialize();

    size_t getSizeByDim(const Dims& dims);

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    void copyInputToDeviceAsync(const cudaStream_t& stream = 0);

    void copyOutputToHostAsync(const cudaStream_t& stream = 0);

    void upscaleDepth(Mat& mask, int targetWidth, int targetHeight, int size);

    void setInput(Mat& image);

private:

    vector<Dims> mInputDims;            //!< The dimensions of the input to the network.
    vector<Dims> mOutputDims;           //!< The dimensions of the output to the network.
    vector<void*> mGpuBuffers;          //!< The vector of device buffers needed for engine execution
    vector<float*> mCpuBuffers;
    vector<size_t> mBufferBindingBytes;
    vector<size_t> mBufferBindingSizes;
    cudaStream_t mCudaStream;

    IRuntime* mRuntime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* mEngine;               //!< The TensorRT engine used to run the network
    IExecutionContext* mContext;        //!< The context for executing inference using an ICudaEngine
};
