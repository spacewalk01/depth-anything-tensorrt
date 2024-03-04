<h1 align="center"><span>TensorRT-YOLOv9 Python</span></h1>

This repo hosts a Python implementation of the YOLOv9 state of the art object detection model, leveraging the TensorRT API for efficient real-time inference.

## Usage 

### Installation

Install the corresponding version of TensorRT API according to your system environment.
```bash
git clone https://github.com/spacewalk01/TensorRT-YOLOv9.git
cd TensorRT-YOLOv9/python
pip install -r requirements.txt
```

### Running

```bash
python yolov9_trt.py --engine <path to trt engine> --data <input dir> --outdir <outdir> 
```

For example:
```bash
python yolov9_trt.py --engine yolov9-c.engine --data images --outdir output
```
