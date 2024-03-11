import os
import torch
import torch.onnx
import argparse

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def export_model(encoder: str, load_from: str, image_shape: tuple):
    """
    Exports a Depth DPT model to ONNX format.

    Args:
        encoder (str): Type of encoder to use ('vits', 'vitb', 'vitl').
        load_from (str): Path to the pre-trained model checkpoint.
        image_shape (tuple): Shape of the input image (channels, height, width).

    Returns:
        str: Path to the exported ONNX model.
    """

    # Initializing model
    assert encoder in ['vits', 'vitb', 'vitl']
    if encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub='localhub')
    elif encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub='localhub')
    else:
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub='localhub')

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    # Loading model weight
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


def main():
    parser = argparse.ArgumentParser(description="Export Depth DPT model to ONNX format")
    parser.add_argument("--encoder", type=str, choices=['vits', 'vitb', 'vitl'], help="Type of encoder to use ('vits', 'vitb', 'vitl')")
    parser.add_argument("--load_from", type=str, help="Path to the pre-trained model checkpoint")
    parser.add_argument("--image_shape", type=int, nargs=3, metavar=("channels", "height", "width"), help="Shape of the input image")
    args = parser.parse_args()

    export_model(args.encoder, args.load_from, tuple(args.image_shape))

if __name__ == "__main__":
    main()
