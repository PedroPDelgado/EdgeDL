import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('C:/Users/dep2brg/Documents/Thesis/Code/data/datasets/vgg16_bn_imagenet_v13.onnx')  # load onnx model
output = prepare(onnx_model)  # run the loaded model
output.export_graph('vgg16_bn_imagenet')  # export the model