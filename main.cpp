#include "depth_anything/trt_module.h"

int main()
{
    string model_path = "./checkpoints/depth_anything_vitb14.onnx";
    string image_path = "your image path";

    TRTModule model(model_path);

    Mat image = imread(image_path);
    Mat depth = model.predict(image);

    Mat colored_depth;
    applyColorMap(depth, colored_depth, COLORMAP_INFERNO);

    imshow("Depth", colored_depth);
    waitKey(0);

    return 0;
}
