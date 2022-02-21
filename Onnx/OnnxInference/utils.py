from asyncore import read
import onnxruntime
import os
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import typing
from onnxruntime.quantization import CalibrationDataReader
import csv


def _get_image_paths(images_dir: str) -> typing.List[str]:
    image_filepaths = []
    dirs = os.listdir(images_dir)
    for dir in dirs:
        curr_dir = images_dir + dir + '/'
        files = os.listdir(curr_dir)
        for file in files:
            image_filepaths.append(curr_dir + file)
    return image_filepaths


def _preprocess_image(img: Image, size: typing.Tuple[int, int]) -> np.array:
    img = img.resize(size)
    data = np.array(img, dtype="float32")
    if not len(data.shape) == 3:
        return None
    data = data.transpose([2, 0, 1]) # HWC -> CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(data.shape[0]):
        data[channel, :, :] = (data[channel, :, :] / 255 - mean[channel]) / std[channel]
    return data


def load_images(images_dir, image_size, max_size, shuffle):
    height, width = image_size
    image_filepaths = _get_image_paths(images_dir)
    if shuffle:
        import random
        random.shuffle(image_filepaths)
    unconcatenated_batch_data = []
    final_image_filepaths = []
    total = min(max_size, len(image_filepaths)) if max_size > 0 else len(image_filepaths)

    for i in range(total):
        img = Image.open(image_filepaths[i])
        image_data = _preprocess_image(img, (height, width))
        if image_data is not None:
            unconcatenated_batch_data.append(image_data)
            final_image_filepaths.append(image_filepaths[i])
    
    batch_data = np.expand_dims(unconcatenated_batch_data, axis=0)
    batch_data = np.vstack(batch_data)
    return batch_data, final_image_filepaths


def create_image_label_dict(labels_filepath):
    labels = {}
    with open(labels_filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            key = row[0].split('/')[-1]
            labels[key] = row[1]
    return labels


def get_model_size(model_path):
    return os.path.getsize(model_path)/(1024*1024)


def get_true_label(image_filepath, images_label_dict):
    image_name = image_filepath.split('/')[-1]
    return images_label_dict[image_name]
    


def run_inference(session: onnxruntime.InferenceSession, inputs: typing.List[np.array], input_name: str) -> dict:
    outputs = []
    times = []

    for i in range(len(inputs)):
        if (i % 50) == 0:
            print(i)
        start = timer()
        pred = session.run([], {input_name: np.expand_dims(inputs[i], axis=0)})
        end = timer()
        times.append(end-start)
        outputs.append(pred[0])
    
    return {'predictions': outputs, 'avg_inference_time': sum(times)/len(times)}


class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, image_size, input_name, max_size=0, shuffle=True):
        self.images_folder = calibration_image_folder
        self.image_height, self.image_width = image_size
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.input_name = input_name
        self.max_size = max_size
        self.shuffle=shuffle

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = load_images(
                images_dir=self.images_folder,
                image_size=(self.image_height, self.image_width),
                max_size=self.max_size,
                shuffle=self.shuffle)[0] # [0] because load_images returns a tuple
                
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{self.input_name: np.expand_dims(nhwc_data, axis=0)} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


