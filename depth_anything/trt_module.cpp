#include "trt_module.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

static Logger gLogger;

#define USE_FP16 true // set USE_FP16 or USE_FP32

TRTModule::TRTModule(string modelPath)
{
    if (getFileExtension(modelPath) == "onnx")
    {
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, USE_FP16);

        auto enginePath = changeFileExtension(modelPath);
        cout << "Saving Engine to " << enginePath << endl;
        saveEngine(enginePath);
    }
    else
    {
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath);
    }
}

TRTModule::~TRTModule()
{
    // Release stream and buffers
    cudaStreamDestroy(mCudaStream);
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    for (int i = 0; i < mCpuBuffers.size(); i++)    
        delete[] mCpuBuffers[i];
    
    // Destroy the engine
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

bool TRTModule::saveEngine(const std::string& fileName) 
{
    if (mEngine)
    {
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "read create engine file" << fileName << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return 1;
}

void TRTModule::build(string onnxPath, bool isFP16)
{
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);

    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);

    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);

    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsed != nullptrt);

    // CUDA stream used for profiling by the builder.
    assert(mCudaStream != nullptr);

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);

    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);
    assert(mEngine != nullptr);

    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    delete network;
    delete config;
    delete parser;
    delete plan;

    initialize();
}

void TRTModule::deserializeEngine(string enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << enginePath << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);
    file.read(serializedEngine, size);
    file.close();

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    assert(*mEngine);
    mContext = mEngine->createExecutionContext();
    assert(*mContext);
    delete[] serializedEngine;

    assert(mEngine->getNbBindings() != inputNames.size() + outputNames.size()); 

    initialize();
}

void TRTModule::initialize()
{
    mGpuBuffers.resize(mEngine->getNbBindings());
    mCpuBuffers.resize(mEngine->getNbBindings());

    for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
    {
        size_t binding_size = getSizeByDim(mEngine->getBindingDimensions(i));
        mBufferBindingSizes.push_back(binding_size);
        mBufferBindingBytes.push_back(binding_size * sizeof(float));

        mCpuBuffers[i] = new float[binding_size];

        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        if (mEngine->bindingIsInput(i))
        {
            mInputDims.push_back(mEngine->getBindingDimensions(i));
        }
        else
        {
            mOutputDims.push_back(mEngine->getBindingDimensions(i));
        }
    }

    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
Mat TRTModule::predict(Mat &inputImage)
{
    const int H = mInputDims[0].d[2];
    const int W = mInputDims[0].d[3];

    auto start_time = std::chrono::high_resolution_clock::now();

    // Preprocessing
    auto resizedImage = resizeImage(inputImage, W, H);
    setInput(resizedImage);

    // Memcpy from host input buffers to device input buffers
    copyInputToDeviceAsync(mCudaStream);

    // Perform inference
    if (!mContext->executeV2(mGpuBuffers.data()))
    {
        cout << "inference error!" << endl;
        return Mat();
    }

    // Memcpy from device output buffers to host output buffers
    copyOutputToHostAsync(mCudaStream);

    // Postprocessing
    Mat outputImage(W, H, CV_32FC1, mCpuBuffers[1]);
    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    upscaleMask(outputImage, inputImage.cols, inputImage.rows, W);

    auto end_time = std::chrono::high_resolution_clock::now();

    // Runtime in microseconds
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Inference time: " << inference_duration.count() << " milliseconds" << std::endl;

    return outputImage;
}

//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void TRTModule::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    memcpyBuffers(true, false, true, stream);
}

//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void TRTModule::copyOutputToHostAsync(const cudaStream_t& stream)
{
    memcpyBuffers(false, true, true, stream);
}

void TRTModule::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        const size_t byteSize = mBufferBindingBytes[i];
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
        {
            if (async)
            {
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}


Mat TRTModule::resizeImage(Mat& img, int inputWidth, int inputHeight)
{
    int w, h;
    float aspectRatio = (float)img.cols / (float)img.rows;

    if (aspectRatio >= 1)
    {
        w = inputWidth;
        h = int(inputHeight / aspectRatio);
    }
    else
    {
        w = int(inputWidth * aspectRatio);
        h = inputHeight;
    }

    Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, INTER_LINEAR);
    Mat out(inputHeight, inputWidth, CV_8UC3, 0.0);
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));

    return out;
}

void TRTModule::upscaleMask(Mat& mask, int targetWidth, int targetHeight, int inputSize)
{
    int limX, limY;
    if (targetWidth > targetHeight)
    {
        limX = inputSize;
        limY = inputSize * targetHeight / targetWidth;
    }
    else
    {
        limX = inputSize * targetWidth / targetHeight;
        limY = inputSize;
    }

    cv::resize(mask(Rect(0, 0, limX, limY)), mask, Size(targetWidth, targetHeight));
}

size_t TRTModule::getSizeByDim(const Dims& dims)
{
    size_t size = 1;
    
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }

    return size;
}

void TRTModule::setInput(Mat& inputImage)
{
    const int inputH = mInputDims[0].d[2];
    const int inputW = mInputDims[0].d[3];
    
    int i = 0;
    for (int row = 0; row < inputImage.rows; ++row)
    {
        uchar* uc_pixel = inputImage.data + row * inputImage.step;
        for (int col = 0; col < inputImage.cols; ++col)
        {
            mCpuBuffers[0][i] = ((float)uc_pixel[2] / 255.0f - 0.485f) / 0.229f;
            mCpuBuffers[0][i + inputImage.rows * inputImage.cols] = ((float)uc_pixel[1] / 255.0f - 0.456f) / 0.224f;
            mCpuBuffers[0][i + 2 * inputImage.rows * inputImage.cols] = ((float)uc_pixel[0] / 255.0f - 0.406f) / 0.225f;
            uc_pixel += 3;
            ++i;
        }
    }
}
