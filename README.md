
<h1 align="center"><span>Depth-Anything TensorRT</span></h1>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repo hosts a C++ and python implementation of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) Monocular Depth Estimation model, leveraging the TensorRT API for efficient real-time inference.
<p align="center" margin: 0 auto;>
  <img src="assets/davis_dolphins_result.gif" height="225px" width="800px" />
</p>
<p align="center">
  <a href="https://github.com/LiheYoung/Depth-Anything/tree/main/assets/examples_video">video link</a> 
</p>

## ⏱️ Performance

The inference time includes the pre-preprocessing and post-processing stages:
| Device          | Model | Model Input (WxH) |  Image Resolution (WxH)     | Inference Time(ms) |
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090        | `Depth-Anything-S`  |518x518  |  1280x720    | 3     |
| RTX4090        | `Depth-Anything-B`  |518x518  |  1280x720    | 6     |
| RTX4090        | `Depth-Anything-L`  |518x518  |  1280x720    | 12     |

Note that the inference was conducted using `FP16` precision, with a warm-up period of 10 frames, and the reported time corresponds to the last inference.

## 🚀 Usage

**Linux:**

``` shell
# infer image
./depth-anything-tensorrt depth_anything_vitb14.engine test.jpg
# infer folder(images)
./depth-anything-tensorrt depth_anything_vitb14.engine data
# infer video
./depth-anything-tensorrt depth_anything_vitb14.engine test.mp4 # the video path
```

**Windows:**

``` shell
# infer image
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.jpg
# infer folder(images)
depth-anything-tensorrt.exe depth_anything_vitb14.engine data
# infer video
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.mp4 # the video path
```

## 🛠️ Setup

Refer to our [docs/INSTALL.md](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/docs/INSTALL.md) for detailed installation instructions.

## 👏 Acknowledgement

This project is based on the following projects:
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
