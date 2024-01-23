
<h1 align="center"><span>NanoSAM C++</span></h1>

This repo provides a TensorRT C++ implementation of [Depth Anything](https://github.com/LiheYoung/Depth-Anything) model, for real-time inference on GPU.

<p align="center" margin: 0 auto;>
  <img src="assets/parkour_merged.gif" height="250px" width="480px" />
</p>

## Getting Started
Load engine or build an engine from onnx and perform depth estimation:

```cpp
TRTModule model("./depth_anything_vitb14.onnx");  // 1. Build an engine from an onnx file
//TRTModule model("./depth_anything_vitb14.engine"); // 2. Load a built engine

Mat image = imread( "./zidan.jpg");

Mat depth = model.predict(image);

Mat colored_depth;
cv::applyColorMap(depth, colored_depth, cv::COLORMAP_INFERNO);

cv::imshow("Depth", colored_depth);
cv::waitKey(0);
```

## Performance
The inference time includes the pre-preprocessing time and the post-processing time:
| Device          | Image Shape(WxH)     | Model Shape(WxH)  | Inference Time(ms) |
|:---------------:|:------------:|:------------:|:------------:|
| RTX4090        |1280x720  |518x5184       | 11     |

## Installation 

1. Install [Depth-Anything model](https://github.com/LiheYoung/Depth-Anything)
```
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```
2. Copy `dpt.py` to `Depth-Anything/depth_anything`. Here I only removed an unfreezing operation at the end of model's forward function to avoid conflicts with tensorrt.
3. Export model to onnx using `export_to_onnx.py`
4. Download the [TensorRT](https://developer.nvidia.com/tensorrt) zip file that matches the Windows version you are using.
5. Choose where you want to install TensorRT. The zip file will install everything into a subdirectory called `TensorRT-8.x.x.x`. This new subdirectory will be referred to as `<installpath>` in the steps below.
6. Unzip the `TensorRT-8.x.x.x.Windows10.x86_64.cuda-x.x.zip` file to the location that you chose. Where:
- `8.x.x.x` is your TensorRT version
- `cuda-x.x` is CUDA version `11.6`, `11.8` or `12.0`
7. Add the TensorRT library files to your system `PATH`. To do so, copy the DLL files from `<installpath>/lib` to your CUDA installation directory, for example, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin`, where `vX.Y` is your CUDA version. The CUDA installer should have already added the CUDA path to your system PATH.
8. Ensure that the following is present in your Visual Studio Solution project properties:
- `<installpath>/lib` has been added to your PATH variable and is present under **VC++ Directories > Executable Directories**.
- `<installpath>/include` is present under **C/C++ > General > Additional Directories**.
- nvinfer.lib and any other LIB files that your project requires are present under **Linker > Input > Additional Dependencies**.
9. Download and install any recent [OpenCV](https://opencv.org/releases/) for Windows.
  
## Acknowledgement
This project is based on the following projects:
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx) - Implementation of popular deep learning networks with TensorRT network definition API.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
