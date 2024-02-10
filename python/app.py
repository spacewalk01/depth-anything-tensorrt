import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import pycuda.driver as cuda  # GPU CPU之间的数据传输
import tensorrt as trt
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from gradio_imageslider import ImageSlider
from PIL import Image
from torchvision.transforms import Compose
import pycuda.autoinit
import time
    

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""

title = "# Depth Anything"
description = """Official demo for **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**.

Please refer to our [paper](https://arxiv.org/abs/2401.10891), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")
    gr.Markdown("You can slide the output to compare the depth prediction with input image")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
    submit = gr.Button("Submit")
        
    def on_submit(image):
        # 创建一个CUDA上下文
        ctx = cuda.Device(0).make_context()
        try:
            original_image = image.copy()

            h, w = image.shape[:2]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = transform({'image': image})['image']
            image = image[None]
            # print(image.shape)
                        
            logger = trt.Logger(trt.Logger.WARNING)
            with open('/home/lwq/DepthAM/depth-anything-tensorrt-build/depth_anything_vitl14.engine', 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            start_time = time.time()
            with engine.create_execution_context() as context:
                input_shape = context.get_tensor_shape('input')
                output_shape = context.get_tensor_shape('output')
                h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
                h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
                d_input = cuda.mem_alloc(h_input.nbytes)
                d_output = cuda.mem_alloc(h_output.nbytes)
                stream = cuda.Stream()
                np.copyto(h_input, image.ravel())
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                depth = h_output
            print(f'Infer time: {int((time.time()-start_time)*1000)}ms')
        
            depth = np.reshape(depth, (518, 518))
            depth = cv2.resize(depth, (w, h))
            raw_depth = Image.fromarray(depth.astype('uint16'))
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            raw_depth.save(tmp.name)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        finally:
            ctx.pop() # 确保在函数结束时弹出上下文
            
        return [(original_image, colored_depth), tmp.name] # colored_depth

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, raw_file])

    example_files = os.listdir('examples')
    example_files.sort()
    example_files = [os.path.join('examples', filename) for filename in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, raw_file], fn=on_submit, cache_examples=False)
    

if __name__ == '__main__':
    demo.queue().launch()