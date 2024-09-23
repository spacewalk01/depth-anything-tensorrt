echo off
cd %~dp0

.\depth-anything-tensorrt.exe -input "%~1" -preview -model depth_anything_v2_vitb.onnx -find-engine -output %~dp1

pause