
<h1 align="center"><span>Depth-Anything TensorRT</span></h1>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repo contains implementations of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) monocular depth estimation model in both C++ and Python. It utilizes the TensorRT API to enable efficient real-time inference.

<p align="center">
  <img src="assets/davis_dolphins_result.gif" height="225px" width="720px" />
</p>

## ⏱️ Performance

The inference time includes the pre-preprocessing and post-processing stages:
| Device          | Model | Model Input (WxH) |  Image Resolution (WxH)     | Inference Time(ms) |
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090        | `Depth-Anything-S`  |518x518  |  1280x720    | 3     |
| RTX4090        | `Depth-Anything-B`  |518x518  |  1280x720    | 6     |
| RTX4090        | `Depth-Anything-L`  |518x518  |  1280x720    | 12     |

> [!NOTE]  
> The inference was conducted using `FP16` precision, with a warm-up period of 10 frames, and the reported time corresponds to the last inference.

## 🚀 Usage

#### C++

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
cd python
python trt_infer.py --engine <path to trt engine> --img <single-img> --outdir <outdir> [--grayscale]
```

## 🛠️ Build

#### C++

Refer to our [docs/INSTALL.md](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/docs/INSTALL.md) for detailed installation instructions.

#### Python

``` shell
cd depth-anything-tensorrt/python
pip install -r requirements.txt
``` 

## 👏 Acknowledgement

This project is based on the following projects:
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
