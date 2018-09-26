# onnx_nnvm
This repo explores converting a caffe2 model to onnx, and then to nnvm.

```caffe2/src/to_onnx_nnvm.py``` firstly converts a caffe2 squeezenet model (downloaded from https://github.com/caffe2/models/tree/master/squeezenet) to onnx. Then it loads the onnx model to caffe2 backend and runs validation on images in ```dataset``` with ```dataset/val.txt``` as lables, to make sure that the accuracy of the onnx model is not lost. Then it converts the onnx model to nnvm, and runs validation on tvm.

Currently, the onnx run on caffe2 backend has a top-1 accuracy of 0.63. However, the compiled nnvm model has a top-1 accuracy of 0.
