###########################################################
# Convert a caffe2 squeezenet to onnx                     #
# caffe2 model is from:                                   #
# https://github.com/caffe2/models/tree/master/squeezenet #
###########################################################

import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

# provide type and shape of the model inputs from value_info
data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 227, 227)
value_info = {
    'data': (data_type, data_shape)
}

predict_net = caffe2_pb2.NetDef()
with open('../caffe2_models/predict_net.pb', 'rb') as f:
    predict_net.ParseFromString(f.read())

# Add this name to avoid the error:
# onnx.onnx_cpp2py_export.checker.ValidationError: \
# Field 'name' of graph is required to be non-empty.
if predict_net.name == "":
    predict_net.name = "squeezenet"

init_net = caffe2_pb2.NetDef()
with open('../caffe2_models/init_net.pb', 'rb') as f:
    init_net.ParseFromString(f.read())

onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
    predict_net,
    init_net,
    value_info,
)

onnx.checker.check_model(onnx_model)

###########################################################
# Load the resulting onnx to caffe2 to make sure          #
# that the conversion succeeds without accuracy loss      #
###########################################################

import onnx
import caffe2.python.onnx.backend

import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import time
import os

# process images
def process_input(url, h, w):
    img = image.load_img(url, target_size=(h, w))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

dataset = '../../dataset'

# load class labels
val = np.loadtxt(os.path.join(dataset, 'val.txt'), dtype=str)

N = 30
height = 227
width = 227

accum_time = 0.
top1_correct_count = 0
top5_correct_count = 0
for i in range(N):
    img_ = process_input(os.path.join(dataset, val[i][0]), \
        height, width).transpose([0, 3, 1, 2])

    start_ = time.time()
    outputs = caffe2.python.onnx.backend.run_model(onnx_model, [img_])
    end_ = time.time()
    accum_time = accum_time + (end_ - start_)

    results = np.asarray(outputs)
    preds = np.squeeze(results)
    preds_lst = list(preds)
    pred_ = preds_lst.index(max(preds_lst))
    if int(val[i][1]) == pred_:
        top1_correct_count = top1_correct_count + 1

top1_accuracy = top1_correct_count * 1.0 / N
throughput = N * 1.0 / accum_time

print('top1 accuracy : ', top1_accuracy)   # 0.63
print('throughput per sec : ', throughput)

###########################################################
# Convert the resulting onnx squeezenet to nnvm           #
###########################################################

import nnvm
import tvm
import numpy as np
sym, params = nnvm.frontend.from_onnx(onnx_model)

import nnvm.compiler
target = 'cuda'
# assume first input name is data
input_name = sym.list_input_names()[0]
print(input_name)  # data_0
shape_dict = {input_name: data_shape}
with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

import os
out_file = '../nnvm_models/squeezenet_from_caffe2_onnx'

if target == 'cuda':
    out_file += '_cuda'

lib_file = out_file + '.so'
graph_file = out_file + '.json'
params_file = out_file + '.params'

lib.export_library(os.path.join('./', lib_file))
with open(os.path.join('./', graph_file), 'w') as fo:
    fo.write(graph.json())
with open(os.path.join('./', params_file), 'wb') as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

###########################################################
# Run validation on TVM                                   #
###########################################################

from tvm.contrib import graph_runtime

# tvm loads pre-compiled model
input_shape = data_shape
output_shape = (1, 1000)
shape_dict = {input_name : input_shape}

ctx = tvm.gpu(0) if target == 'cuda' else tvm.cpu(0)
loaded_lib = tvm.module.load(lib_file)
loaded_graph = open(graph_file).read()
loaded_params = bytearray(open(params_file, "rb").read())
params = nnvm.compiler.load_param_dict(loaded_params)
module = graph_runtime.create(loaded_graph, loaded_lib, ctx)

N = 30
accum_time = 0.
top1_correct_count = 0
top5_correct_count = 0
for i in range(N):
    img_ = process_input(os.path.join(dataset, val[i][0]), \
        height, width).transpose([0, 3, 1, 2])

    module.set_input(**params)
    start_ = time.time()
    module.set_input(input_name, tvm.nd.array(img_.astype('float32')))
    module.run()
    out = module.get_output(0, tvm.nd.empty(output_shape, 'float32'))
    end_ = time.time()
    accum_time = accum_time + (end_ - start_)

    tvm_out = out.asnumpy()
    if int(val[i][1]) == tvm_out.argmax():
        top1_correct_count = top1_correct_count + 1
    if int(val[i][1]) in (-tvm_out).argsort()[0][:5]:
        top5_correct_count = top5_correct_count + 1

top1_accuracy = top1_correct_count * 1.0 / N
top5_accuracy = top5_correct_count * 1.0 / N
throughput = N * 1.0 / accum_time

print('top1 accuracy : ', top1_accuracy)
print('top5 accuracy : ', top5_accuracy)
print('throughput per sec : ', throughput)

# img prediction
for i in range(5):
    img_ = process_input(os.path.join(dataset, val[i][0]), \
        height, width).transpose([0, 3, 1, 2])
    module.set_input(**params)
    module.set_input(input_name, tvm.nd.array(img_.astype('float32')))
    module.run()
    out = module.get_output(0, tvm.nd.empty(output_shape, 'float32'))
    tvm_out = out.asnumpy()
    pred = tvm_out.argmax()
    print('img ' + str(i))
    print('prediction is ' + str(pred))
    print('should be ' + val[i][1])
