
<h1 align="center"><span>Depth-Anything TensorRT C++</span></h1>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repository contains a C++ implementation of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) model using the TensorRT API for real-time inference.

<p align="center" margin: 0 auto;>
  <img src="assets/parkour_merged.gif" height="225px" width="800px" />
</p>

## Performance
The inference time includes the pre-preprocessing and post-processing stages:
| Device          | Model | Model Input (WxH) |  Image Resolution (WxH)     | Inference Time(ms) |
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090        | `Depth-Anything-S`  |518x518  |  1280x720    | 3     |
| RTX4090        | `Depth-Anything-B`  |518x518  |  1280x720    | 6     |
| RTX4090        | `Depth-Anything-L`  |518x518  |  1280x720    | 12     |

Note that the inference was conducted using `FP16` precision, with a warm-up period of 10 frames, and the reported time corresponds to the last inference.

- ➡️ 🍼 [How to use this icons?](#-how-to-use-it)

- [Setup Guide](https://github.com/spacewalk01/depth-anything-tensorrt/issues/10) 🚀 


## Acknowledgement
This project is based on the following projects:
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
