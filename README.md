<div align="left">

Depth Anything TensorRT
===========================

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-11.8-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/LICENSE)

</div>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repo provides a TensorRT implementation of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) depth estimation model in both C++ and Python, enabling efficient real-time inference.

<p align="center">
  Depth-Anything-V1
  <img src="assets/davis_dolphins_result.gif" height="225px" width="720px" />
</p>
<p align="center">
  Depth-Anything-V2
  <img src="assets/ferris_wheel_result.gif" height="225px" width="720px" />
</p>



## News

* **2024-06-17:** [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) is integrated.
* **2024-01-23:** Depth Anything [TensorRT](https://github.com/spacewalk01/depth-anything-tensorrt) version is created.
  
## ⏱️ Performance

The inference time includes the pre-preprocessing and post-processing stages:
| Device          | Model | Model Input (WxH) |  Image Resolution (WxH)|Inference Time(ms)|
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090        | Depth-Anything-S  |518x518  |  1280x720    | 3     |
| RTX4090        | Depth-Anything-B  |518x518  |  1280x720    | 6     |
| RTX4090        | Depth-Anything-L  |518x518  |  1280x720    | 12    |


> [!NOTE]
> Inference was conducted using `FP16` precision, with a warm-up period of 10 frames. The reported time corresponds to the last inference.

## 🚀 Quick Start

#### C++

- **Usage 1**: Create an engine from an onnx model and save it:
``` shell
depth-anything-tensorrt.exe <onnx model> <input image or video>
```
- **Usage 2**: Deserialize an engine. Once you've built your engine, the next time you run it, simply use your engine file:
``` shell
depth-anything-tensorrt.exe <engine> <input image or video>
```

Example:
``` shell
# infer image
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.jpg
# infer folder(images)
depth-anything-tensorrt.exe depth_anything_vitb14.engine data
# infer video
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.mp4 # the video path
```

#### Python

```
cd depth-anything-tensorrt/python

# infer image
python trt_infer.py --engine <path to trt engine> --img <single-img> --outdir <outdir> [--grayscale]
```

## 🛠️ Build

#### C++

Refer to our [docs/INSTALL.md](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/docs/INSTALL.md) for C++ environment installation.

#### Python

``` shell
cd <tensorrt installation path>/python
pip install cuda-python
pip install tensorrt-8.6.0-cp310-none-win_amd64.whl
pip install opencv-python
``` 

## 🤖 Model Preparation
### Depth-Anything-V1
Perform the following steps to create an onnx model:

1. Download the pretrained [model](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) and install [Depth-Anything](https://github.com/LiheYoung/Depth-Anything):
   ``` shell
   git clone https://github.com/LiheYoung/Depth-Anything
   cd Depth-Anything
   pip install -r requirements.txt
   ```

2. Copy and paste all files in depth-anything_v1 to `<depth_anything_installpath>/depth_anything` folder. Note that I've only removed a squeeze operation at the end of model's forward function in `dpt.py` to avoid conflicts with TensorRT.
3. Export the model to onnx format using [export.py](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/export.py). You will get an onnx file named `depth_anything_vit{}14.onnx`, such as `depth_anything_vitb14.onnx`. Note that I used torch cpu version for exporting the onnx model as it is not necessary to deploy the model on GPU when exporting.

    
    ``` shell
    conda create -n depth-anything python=3.8
    conda activate depth-anything
    pip install torch torchvision
    pip install opencv-python
    pip install onnx
    python export_v1.py --encoder vitb --load_from depth_anything_vitb14.pth --image_shape 3 518 518
    ```


### Depth-Anything-V2

1. Clone [Depth-Anything-V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/tree/main) 
   ``` shell
   git clone https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
   cd Depth-Anything-v2
   pip install -r requirements.txt
   ```
2. Download the pretrained models from the [readme](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/README_Github.md) and put them in checkpoints folder:
3. Copy and paste all files in depth_anything_v2 to `<depth_anything_installpath>/depth_anything_v2` folder. 
4. Run the following to export the model:
    ``` shell
    conda create -n depth-anything python=3.8
    conda activate depth-anything
    pip install torch torchvision
    pip install opencv-python
    pip install onnx
    python export_v2.py --encoder vitb --input-size 518
    ```

> [!TIP]
> The width and height of the model input should be divisible by 14, the patch height.
    
## 👏 Acknowledgement

This project is based on the following projects:
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
