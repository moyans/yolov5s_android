import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision

from utils import non_max_suppression
from detector_head import Detect
from PIL import Image

class TfLiteRunner():
    def __init__(
        self, 
        model_path, 
        input_size=640, 
        conf_thres=0.25, 
        iou_thres=0.45, 
        quantize_mode=False, 
        is_output_quantized=False,
        class_nums = 1,
    ):

        self.INPUT_SIZE = input_size
        self.interpreter = tf.lite.Interpreter(model_path, num_threads=8)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_tensor_names = [
            'Identity',
            'Identity_1',
            'Identity_2'
        ]

        # reorder output details because sometimes output_details() order is unintended order.
        output_details = self.interpreter.get_output_details()
        self.output_details = [
            next(filter(lambda detail: detail['name'] == name, output_details), None)
            for name in self.output_tensor_names
        ]
        assert (None not in self.output_details), "model does not contain specified 'output_tensor_names'  "

        self.input_shape = self.input_details[0]['shape']
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.quantize_mode = quantize_mode
        self.is_output_quantized = is_output_quantized
        self.stride = [8, 16, 32]
        self.output_shape = [int(input_size / xt) for xt in self.stride]

        self.detector = Detect(nc=class_nums, input_size=input_size, stride=self.stride)


    def run_tflite(self, input_data):
        assert(isinstance(input_data, np.ndarray))
        assert(input_data.shape == (1, self.INPUT_SIZE, self.INPUT_SIZE, 3))
        
        input_dtype = np.uint8 if self.quantize_mode else np.float32 
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(input_dtype))
        self.interpreter.invoke()

        output_datas = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])

            if self.is_output_quantized:
                # dequantize
                scale, zero_point = output_detail['quantization']
                print(f'scale = {scale}, zero_point = {zero_point}')
                output_data = output_data.astype(np.float32)
                output_data = (output_data - zero_point) * scale
            output_datas.append(output_data)

        return output_datas

    def preprocess(self, img_path):
        input_data = Image.open(img_path)
        org_w, org_h = input_data.size
        self.ratio_h = org_h/float(self.INPUT_SIZE) 
        self.ratio_w = org_w/float(self.INPUT_SIZE)

        # PIL open image only resize
        input_data = input_data.resize((self.INPUT_SIZE, self.INPUT_SIZE))
        input_data = np.array(input_data)
        input_data = input_data[np.newaxis, ...]

        if not self.quantize_mode:
            preprocessed = input_data / 255.0
        else:
            scale, zero_point = self.input_details[0]['quantization']
            # scale is always 1/255, and zero_point is always 0
            print(f'scale = {scale}, zero_point = {zero_point}')
            input_data = input_data / 255.0
            input_data = input_data / scale + zero_point
            preprocessed = input_data.astype(np.uint8)

        print("input tensor shape: ", preprocessed.shape)
        return preprocessed

    def postprocess(self, detector_out):
        bboxres = non_max_suppression(detector_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        bboxres[0][:, 0] *= self.ratio_w
        bboxres[0][:, 2] *= self.ratio_w
        bboxres[0][:, 1] *= self.ratio_h
        bboxres[0][:, 3] *= self.ratio_h
        return bboxres

    def detect(self, img_path):
        preprocessed = self.preprocess(img_path)
        tflite_out = self.run_tflite(preprocessed)
        tflite_out = [torch.from_numpy(x) for x in tflite_out]
        detector_out = self.detector(tflite_out)
        bboxres = self.postprocess(detector_out)
        return bboxres