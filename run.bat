echo off
echo %~1

.\"depth-anything-tensorrt.exe" -input "%~1" -preview -model depth_anything_v2_vitb.onnx -find-engine

pause