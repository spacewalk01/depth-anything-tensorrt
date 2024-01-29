cmake_minimum_required(VERSION 3.12)
project(depth-anything-tensorrt-simplified)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SOURCES
    main.cpp
    inference.cpp
	utils.cpp
	depth_anything.cpp
    # ... add more source files here if needed
)

# Add headers
set(HEADERS
    inference.h
	utils.h
	depth_anything.h
    # ... add more header files here if needed
)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
set(OpenCV_DIR "D:\\opencv\\build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "D:\\0_dp\\library\\TensorRT-8.4.3.1")

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)