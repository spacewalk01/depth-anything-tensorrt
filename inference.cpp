#include "inference.h"

cv::Mat inference(cv::Mat& image,  DepthAnything& depth_model)
{
	cv::Mat im0,im1;
	image.copyTo(im0);
	// Inference depth model
	im1 = depth_model.predict(im0);

	return im1;
}
