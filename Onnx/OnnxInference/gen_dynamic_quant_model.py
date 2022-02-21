from onnxruntime.quantization import quantize_dynamic, QuantType


"""
documentation available at: https://onnxruntime.ai/docs/performance/quantization.html
"""


def dynamic_quantization(fp32_model : str, save_path: str):
    quantize_dynamic(fp32_model, save_path, activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)


if __name__ == '__main__':
    
    # dynamic quantization
    quantized_model = dynamic_quantization(
        fp32_model='C:/Users/dep2brg/Documents/Thesis/Code/data/models/mobilenetv3_large_imagenet_v14.onnx',
        save_path='C:/Users/dep2brg/Documents/Thesis/Code/data/models/mobilenetv3_large_imagenet_v14_dynamicquant.onnx')

