#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "inference.h"
#include "depth_anything.h"

using namespace std;
/* 1 for video, 0 for folder*/
#define FileFormat 0 
// path to DepthAnything engine
#define depthEngineFile "depth_anything_vitb14.engine"

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main()
{
    // init model
    DepthAnything depth_model(depthEngineFile, logger);

#if FileFormat
    //path to video
    string VideoPath = "wuhan_day.avi";
    // open cap
    cv::VideoCapture cap(VideoPath);

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Create a VideoWriter object to save the processed video
    cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
    while (1)
    {
        cv::Mat frame;
        cv::Mat show_frame;
        cap >> frame;

        if (frame.empty())
            break;
        frame.copyTo(show_frame);
        cv::Mat new_frame;
        frame.copyTo(new_frame);
        auto start = std::chrono::system_clock::now();
        cv::Mat result = inference(frame, depth_model);
        auto end = std::chrono::system_clock::now();
        std::cout << "Time of per frame:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        cv::addWeighted(show_frame, 0.7, result, 0.3, 0.0, show_frame);
        imshow("depth_result", result);
        imshow("full_result", show_frame);
        cv::waitKey(100);
        if (cv::waitKey(1) == 'q')
            break;
    }

    // Release resources
    cv::destroyAllWindows();
    cap.release();
    output_video.release();
#else
    // path to folder containing images
    string imageFolderPath = "mytest/";

    // Get the list of files in the folder
    vector<std::string> imageFiles;
    cv::glob(imageFolderPath + "/*.jpg", imageFiles, false);

    // path to folder saves images
    string imageFolderPath_out = "mytest_out/";
    for (const auto& imagePath : imageFiles)
    {
        // open image
        cv::Mat frame = cv::imread(imagePath);
        std::cout << "imagePath" << imagePath  << std::endl;
        if (frame.empty())
        {
            cerr << "Error reading image: " << imagePath << endl;
            continue;
        }

        cv::Mat show_frame;
        frame.copyTo(show_frame);

        auto start = chrono::system_clock::now();
        cv::Mat result = inference(frame, depth_model);
        auto end = chrono::system_clock::now();
        cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        addWeighted(show_frame, 0.7, result, 0.3, 0.0, show_frame);
        imshow("depth_result", result);
        imshow("full_result", show_frame);
        cv::waitKey(100);

        std::istringstream iss(imagePath);
        std::string token;
        // 使用 '/' 作为分隔符
        while (std::getline(iss, token, '/')) 
        {  

        }
        cv::imwrite(imageFolderPath_out+token, result);
    }
#endif
    return 0;
}

