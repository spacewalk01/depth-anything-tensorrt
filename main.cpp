#include <iostream>
#include <string>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "depth_anything.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

// Helper function to replace all occurrences of a character in a string
void replaceChar(std::string& str, char find, char replace) {
    size_t pos = 0;
    while ((pos = str.find(find, pos)) != std::string::npos) {
        str[pos] = replace;
        pos++;
    }
}

bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

bool createFolder(const std::string& folderPath) {
#ifdef _WIN32
    if (!CreateDirectory(folderPath.c_str(), NULL)) {
        DWORD error = GetLastError();
        if (error == ERROR_ALREADY_EXISTS) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        } else {
            std::cerr << "Failed to create folder! Error code: " << error << std::endl;
            return false; // Failed to create folder
        }
    }
#else
    if (mkdir(folderPath.c_str(), 0777) != 0) {
        if (errno == EEXIST) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        } else {
            std::cerr << "Failed to create folder! Error code: " << errno << std::endl;
            return false; // Failed to create folder
        }
    }
#endif
    std::cout << "Folder created successfully!" << std::endl;
    return true; // Folder created successfully
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv) {
    bool model_loaded = false;

    // all valid video & image extensions
    vector<string> video_extensions = {"avi", "mp4", "m4v", "mpeg", "mov", "mkv"};
    vector<string> image_extensions = {"jpeg", "jpg", "png"};

    // organizing options & arguements into map
    unordered_map<string, string> options;
    string previous_option = "";
    string current_arguement = "";
    DepthAnything depth_model;
    int cutoff = 0;
    for (int i = 0; i < argc; i++) {
        cutoff = 0;
        current_arguement = argv[i];
        if (current_arguement[0] == '-') {
            if (current_arguement[1] == '-') {
                cutoff = 2;
            } else {
                cutoff = 1;
            }
            previous_option = current_arguement.substr(cutoff);
            options[previous_option] = "1";
        } else if (!previous_option.empty()) {
            options[previous_option] = current_arguement.substr(cutoff);
            previous_option = "";
        }
    }

    // load selected model
    if (!options["model"].empty()) {
        string model_path = options["model"];

        if (!IsFile(model_path)) {
            cout << "Model not found!" << endl;
            abort();
        }
        cout << "Loading model: \"" << model_path << "\"" <<  endl;

        string alternate_path = model_path.substr(0, model_path.length() - 5) + ".engine";
        if (model_path.substr(model_path.find_last_of('.') + 1) == "onnx" && IsFile(alternate_path)) {
            string confirm_engine = "";
            cout << "\"" << alternate_path << "\" found, Override existing .engine file? (Y/N): ";
            cin >> confirm_engine;
            if (confirm_engine != "Y" && confirm_engine != "y"){
                model_path = alternate_path;
            }
        }
        depth_model.init(model_path, logger);
        cout << "Model successfully loaded." << endl << endl;
        model_loaded = true;
    }

    if (!options["input"].empty()) {
        string input = options["input"];
        std::vector<std::string> imagePathList;
        std::vector<std::string> videoPathList;
        string suffix = input.substr(input.find_last_of('.') + 1);
        bool suffix_found = false;
        // organize images and videos in path into seperate lists
        if (IsFile(input)) {
            for (string& proper_suffix : image_extensions) {
                if (suffix == proper_suffix) {
                    imagePathList.push_back(input);
                    suffix_found = true;
                    break;
                }
            }
            if (!suffix_found) {
                for (string& proper_suffix : video_extensions) {
                    if (suffix == proper_suffix) {
                        videoPathList.push_back(input);
                        suffix_found = true;
                        break;
                    }
                }
            }
            if (!suffix_found) {
                printf("Incorrect suffix %s!\n", suffix.c_str());
                std::abort();
            }
        } else if (IsPathExist(input)) {
            vector<string> current_extension_found;
            for (string& current_extension : image_extensions) {
                cv::glob(input + "/*." + current_extension, current_extension_found);
                imagePathList.insert(
                    imagePathList.end(),
                    current_extension_found.begin(),
                    current_extension_found.end()
                );
                current_extension_found.clear();
            }
            for (string& current_extension : video_extensions) {
                cv::glob(input + "/*." + current_extension, current_extension_found);
                videoPathList.insert(
                    videoPathList.end(),
                    current_extension_found.begin(),
                    current_extension_found.end()
                );
                current_extension_found.clear();
            }
        } else {
            cout << "Input location invalid!" << endl;
        }


        if (model_loaded) {
            string base_filename;
            string filename;
            string prefix = "depth_";
            string output;
            string generated_name;
            string full_output_location;
            string current_suffix;
            if (!options["prefix"].empty()) {
                prefix = options["prefix"];
            }
            //iterate through videoPathList and render depthmaps.
            for (const auto& videoPath : videoPathList) {
                base_filename = videoPath.substr(videoPath.find_last_of("/\\") + 1);
                filename = base_filename.substr(0, base_filename.find_last_of('.'));
                generated_name = prefix + filename;

                if (!options["output"].empty()) {
                    output = options["output"];
                    if (output.back() != '\\') {
                        output += "\\";
                    }
                    if (!IsPathExist(output)) {
                        cout << "Output location invalid!" << endl;
                        abort();
                    }
                }
                full_output_location = output + generated_name;

                // open cap
                cv::VideoCapture cap(videoPath);

                int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
                int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                double fps;
                double frame_total = cap.get(cv::CAP_PROP_FRAME_COUNT);
                int frame_num = 0;
                double previous_tpf = 0;
                double estimate = 0;
                double tpf = 0;
                int estimate_rounded = 0;
                int progress = 0;

                if (!options["fps"].empty()) {
                    try {
                        fps = stod(options["fps"]);
                    } catch (const invalid_argument&) {
                        cerr << "Invalid fps value!" << endl;
                        abort();
                    }
                } else {
                    fps = cap.get(cv::CAP_PROP_FPS);
                }

                // Create a VideoWriter object to save the processed video
                full_output_location += ".avi";

                cout << full_output_location << ":" << endl;
                cv::VideoWriter output_video(full_output_location, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
                while (1) {
                    frame_num ++;
                    cv::Mat frame;
                    cap >> frame;

                    if (frame.empty())
                        break;
                    auto start = std::chrono::system_clock::now();
                    cv::Mat result_d = depth_model.predict(frame);
                    auto end = chrono::system_clock::now();
                    tpf = chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

                    if (previous_tpf == 0) {
                        estimate = (frame_total - frame_num) * (tpf / 1000.0);
                    } else if (frame_num <= 10) {
                        estimate = (frame_total - frame_num) * (((tpf * (frame_num - 1) + previous_tpf) / frame_num) / 1000.0);
                    } else {
                        estimate = (frame_total - frame_num) * (((tpf * (9) + previous_tpf) / 10) / 1000.0);
                    }
                    previous_tpf = tpf;
                    progress = floor((frame_num / frame_total) * 100);
                    cout << "frame#:" << setw(6) << frame_num << " progress:" << setw(3) << progress << "% time per frame:" << setw(9) << tpf << "ms fps:" << setw(4) << floor(100 / (tpf / 1000)) / 100.0 << " eta:";

                    if (estimate >= 60) {
                        estimate_rounded = estimate;
                        cout << setw(3) << floor(estimate / 60) << ":" << setw(2) << estimate_rounded % 60;
                    } else {
                        cout << setw(5) << floor(estimate * 100) / 100.0 << "sec";
                    }

                    cout << " [" << string(floor(progress / 2.5), 'I') << string(40 - floor(progress / 2.5), ' ') << "]";

                    if (options["one-line"].empty()) {
                        cout << endl;
                    } else {
                        cout << setw(20) << "  \r";
                    }
                    if (!options["preview"].empty()) {
                        cv::Mat show_frame;
                        frame.copyTo(show_frame);
                        cv::Mat result;
                        cv::hconcat(show_frame, result_d, result);
                        cv::resize(result, result, cv::Size(width, height / 2)); //IMPORTANT
                        imshow("Depth: Before -> After", result);
                    }
                    output_video.write(result_d);
                    cv::waitKey(1);
                }
                if (!options["one-line"].empty()) {
                    cout << endl;
                }
                // Release resources
                cv::destroyAllWindows();
                cap.release();
                output_video.release();
                cout << full_output_location << " finished generating." << endl << endl;
            }
            // iterate through imagePathList and render depthmaps.
            for (const auto& imagePath : imagePathList) {
                
                base_filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);
                filename = base_filename.substr(0, base_filename.find_last_of('.'));
                generated_name = prefix + filename;

                if (!options["output"].empty()) {
                    output = options["output"];
                    if (output.back() != '\\') {
                        output += "\\";
                    }
                    if (!IsPathExist(output)) {
                        cout << "Output location invalid!" << endl;
                        abort();
                    }
                }

                current_suffix = imagePath.substr(imagePath.find_last_of('.') + 1);

                full_output_location = output + generated_name + "." + current_suffix;

                cout << full_output_location << ":" << endl;

                // open image
                cv::Mat frame = cv::imread(imagePath);
                if (frame.empty())
                {
                    cerr << "Error reading image: " << imagePath << endl;
                    continue;
                }
                

                auto start = chrono::system_clock::now();
                cv::Mat result_d = depth_model.predict(frame);
                auto end = chrono::system_clock::now();
                double tpf = chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
                cout << "time per frame:" << setw(9) << tpf << "ms fps:" << setw(4) << floor(100 / (tpf / 1000)) / 100.0 << endl;
                if (!options["preview"].empty()) {
                    cv::Mat show_frame;
                    frame.copyTo(show_frame);
                    cv::Mat result;
                    cv::hconcat(show_frame, result_d, result);
                    cv::resize(result, result, cv::Size(1080, 480));
                    imshow("depth_result", result);
                    cv::waitKey(1);
                }
                
                cv::imwrite(full_output_location, result_d);

                cout << full_output_location << " finished generating." << endl << endl;
            }
        }
    }
    return 0;
}
