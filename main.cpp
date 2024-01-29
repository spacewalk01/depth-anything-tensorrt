#include "depth_anything/trt_module.h"

int main()
{
    string model_path = "./depth_anything_vitb14.engine";
    string image_path = "./zidane.jpg";

    TRTModule model(model_path);

    Mat image = imread(image_path);
    Mat depth = model.predict(image);

    Mat colored_depth;
    applyColorMap(depth, colored_depth, COLORMAP_INFERNO);

    imshow("Depth", colored_depth);
    waitKey(0);

    return 0;
}
