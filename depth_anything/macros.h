#ifndef __MACROS_H
#define __MACROS_H

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

std::string changeFileExtension(const std::string& fileName) {
    // Find the position of the last '.' in the file name
    size_t dotPosition = fileName.find_last_of('.');

    // Check if a dot was found
    if (dotPosition != std::string::npos) {
        // Create the new file name with .engine extension
        return fileName.substr(0, dotPosition) + ".engine";
    }
    else {
        // Return the original file name if there is no dot
        std::cerr << "Error: Invalid file name format." << std::endl;
        return fileName;
    }
}

std::string getFileExtension(const std::string& filePath) {
    size_t dotPos = filePath.find_last_of(".");
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    return ""; // No extension found
}

#endif  // __MACROS_H
