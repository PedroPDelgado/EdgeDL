import onnxruntime
import numpy as np
from imagenet_labels import labels
import utils
import re


NUM_IMAGES = 0
VAL_DATA_PATH = 'C:/Users/dep2brg/Documents/Thesis/Code/data/datasets/FastAI_ImageNet_v2/val/'
ONNX_MODEL_PATH = 'C:/Users/dep2brg/Documents/Thesis/Code/data/models/mobilenetv3_large_imagenet_v14_staticquant.onnx'
LABELS_FILEPATH = 'C:/Users/dep2brg/Documents/Thesis/Code/data/datasets/FastAI_ImageNet_v2/Validation.csv'


def match(prediction, true_label):
    prediction = re.sub('\.', '', prediction)
    list_predictions = re.split(',\s', prediction)
    for pred in list_predictions:
        pred = pred.lower()
        pred = re.sub('\s', '_', pred)
        if pred == true_label:
            return True
    return False


print("Loading images...")
images, image_filepaths = utils.load_images(images_dir=VAL_DATA_PATH, image_size=(224, 224), max_size=NUM_IMAGES, shuffle=True)
print(f"Loaded {len(images)} images.")
images2label = utils.create_image_label_dict(labels_filepath=LABELS_FILEPATH)
print("Starting inference...")
session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, None)
input_name = session.get_inputs()[0].name
results = utils.run_inference(session=session, inputs=images, input_name=input_name)

true_count = 0
for i in range(len(results['predictions'])):
    prediction = labels[np.argmax(results['predictions'][i], axis=1)[0]]
    true_label = utils.get_true_label(image_filepath=image_filepaths[i], images_label_dict=images2label)
    is_match = match(prediction, true_label)
    if is_match:
        true_count += 1


total_images = len(results['predictions'])
model_acc = true_count/total_images
avg_inference_time = results['avg_inference_time']
model_size = utils.get_model_size(model_path=ONNX_MODEL_PATH)

print(f"Model accuracy {model_acc * 100} %.")
print(f'Average inference time {avg_inference_time} seconds. (Total of {total_images} images)')
print(f"Model size {model_size} MB.")
