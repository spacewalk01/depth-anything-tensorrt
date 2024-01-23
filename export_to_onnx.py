import cv2
import numpy as np
import os
import torch
import torch.onnx

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

encoder = 'vitb'
load_from = './checkpoints/depth_anything_vitb14.pth'
image_shape = (3, 518, 518)

assert encoder in ['vits', 'vitb', 'vitl']
if encoder == 'vits':
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub='localhub')
elif encoder == 'vitb':
    depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub='localhub')
else:
    depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub='localhub')

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'), strict=True)

depth_anything.eval()

# Define dummy input data
dummy_input = torch.ones(image_shape).unsqueeze(0)

# Provide an example input to the model, this is necessary for exporting to ONNX
example_output = depth_anything(dummy_input)

onnx_path = load_from.split('/')[-1].split('.pth')[0] + '.onnx'

# Export the PyTorch model to ONNX format
torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)

print(f"Model exported to {onnx_path}")