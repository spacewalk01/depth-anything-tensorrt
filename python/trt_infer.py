#导入必用依赖
import cv2
import numpy as np
import pycuda.autoinit  # 负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  # GPU CPU之间的数据传输
import tensorrt as trt
from depth_anything.util.transform import load_image

input_image, ori_shape = load_image('/home/lwq/DepthAM/ori_img/test.jpg')

#创建logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)
#创建runtime并反序列化生成engine
with open('/home/lwq/DepthAM/depth-anything-tensorrt/depth_anything_vitl14.engine', 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    input_shape = context.get_tensor_shape('input')
    output_shape = context.get_tensor_shape('output')
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    np.copyto(h_input, input_image.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    depth = h_output
    
# 将depth转换为图片
depth = np.reshape(depth, (518, 518))
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
cv2.imwrite(f'test.jpg', depth)
