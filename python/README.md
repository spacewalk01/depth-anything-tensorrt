<h1 align="center"><span>Depth-Anything TensorRT Python</span></h1>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repo hosts a Python implementation of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) Monocular Depth Estimation model, leveraging the TensorRT API for efficient real-time inference.


## Usage 

### Installation

Install the corresponding version of TensorRT API according to your system environment.
```bash
git clone https://github.com/spacewalk01/depth-anything-tensorrt.git
cd depth-anything-tensorrt/python
pip install -r requirements.txt
```

### Running

```bash
python trt_infer.py --engine <path to trt engine> --img <single-img> --outdir <outdir> [--grayscale]
```

For example:
```bash
python trt_infer.py --engine path/to/engine --img examples/demo1.png --outdir depth_vis
```

### Gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 

To use gradio demo locally, you shoud set `engine_path` in `app.py` then:

```bash
python app.py
```
