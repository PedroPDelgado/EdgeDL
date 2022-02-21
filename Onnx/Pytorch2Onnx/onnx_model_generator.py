import torch
import torch.onnx
from torchvision import models as models
from torchvision.models.vgg import cfgs, VGG, make_layers
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf


VERSION = 14
MODEL_OUTPUT_NAME = 'mobilenetv3_large_imagenet'


def get_model(model_name, pretrained_model_path):
    
    #model = models.mobilenet_v3_large(pretrained=True)
    # can't directly get model from url...
    if model_name == 'vgg_16_bn':
        model = VGG(make_layers(cfgs['D'], batch_norm=True))
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
    elif model_name == 'mobilenetv3_large':
        arch = "mobilenet_v3_large"
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
        model = MobileNetV3(inverted_residual_setting, last_channel)
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
    return model


def export_onnx(model, input_size, save_path, opset_version, input_names, output_names, dynamic_axes, clip_interval=None):
    
    model.eval() # turn the model to inference mode

    dummy_input = torch.randn(1, *input_size, requires_grad=True) # Because export needs to run the model

    if clip_interval is not None:
        torch.clip(input=dummy_input, min=clip_interval[0], max=clip_interval[1])

    # Export the model   
    torch.onnx.export(model,            # model being run 
         dummy_input,                   # model input (or a tuple for multiple inputs) 
         save_path,                     # where to save the model  
         export_params=True,            # store the trained parameter weights inside the model file 
         opset_version=opset_version,   # the ONNX version to export the model to 
         do_constant_folding=True,      # whether to execute constant folding for optimization 
         input_names = input_names,     # the model's input names 
         output_names = output_names,   # the model's output names 
         dynamic_axes=dynamic_axes)     # variable length axes 

    print(f"Model exported to {save_path}")


if __name__ == '__main__':
    
    model = get_model(model_name='mobilenetv3_large', pretrained_model_path='C:/Users/dep2brg/Documents/Thesis/Code/data/models/mobilenet_v3_large-8738ca79.pth')

    export_onnx(
        model=model,
        input_size=(3, 224, 224),
        save_path=MODEL_OUTPUT_NAME + f'_v{VERSION}.onnx',
        opset_version=VERSION,
        input_names=['modelInput'],
        output_names=['modelOutput'],
        dynamic_axes={'modelInput' : {0 : 'batch_size'},
                                'modelOutput' : {0 : 'batch_size'}},
        clip_interval=(0, 1)
    )
