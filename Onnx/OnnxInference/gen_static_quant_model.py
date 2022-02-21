import onnxruntime
from utils import MobilenetDataReader
from onnxruntime.quantization import quantize_static
from onnxruntime.quantization import CalibrationMethod

"""
See: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/imagenet_v2/mobilenet.ipynb

During the calibration process, a model tends to overfit the dataset its being calibrated on. To avoid overfitting, use separate datasets for calibration and validation.

Interesting read on different approach to PTQ using calib data: http://proceedings.mlr.press/v139/hubara21a/hubara21a.pdf

Interesting read on type of calib data: https://arxiv.org/abs/2105.07331
"""


CALIB_DATA = 'C:/Users/dep2brg/Documents/Thesis/Code/data/datasets/FastAI_ImageNet_v2/train/'
FP32_MODEL_PATH = 'C:/Users/dep2brg/Documents/Thesis/Code/data/models/mobilenetv3_large_imagenet_v14.onnx'
SAVE_QUANT_PATH = 'C:/Users/dep2brg/Documents/Thesis/Code/data/models/'

session = onnxruntime.InferenceSession(FP32_MODEL_PATH, None)
input_name = session.get_inputs()[0].name
dr = MobilenetDataReader(calibration_image_folder=CALIB_DATA, image_size=(224, 224), input_name=input_name, max_size=200, shuffle=True)
quantize_static(
    model_input=FP32_MODEL_PATH, model_output=SAVE_QUANT_PATH + 'mobilenetv3_large_imagenet_v14_staticquant.onnx',
    calibration_data_reader=dr)
