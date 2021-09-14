import argparse
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import tensorflow_datasets as tfds

def load_from_cv(img_path, img_size=(640, 640), stride=32, auto=False):
    img0 = cv2.imread(img_path)  # BGR
    # short size resize_paddingg
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def quantize_model(INPUT_SIZE, pb_path, output_path, calib_num, tfds_root, download_flag):
    
    def representative_dataset_gen(img_dir='/d/Dataset/widerface/images/test'):
	    img_list = os.listdir(img_dir)
	    for i in range(200):
	        img_name = img_list[i]
	        img_path = os.path.join(img_dir, img_name)
	        assert os.path.exists(img_path)
	        img = load_from_cv(img_path)
	        img = img.astype('float32')
	        img = img / 255.0 
	        img = img.transpose(1, 2, 0)
	        if len(img.shape) == 3: img = img[None]
	        print('input tensor: ', img.shape)
	        yield [img.astype(np.float32)]


    input_arrays = ['inputs']
    output_arrays = ['Identity', 'Identity_1', 'Identity_2']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)
    converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = False
    converter.inference_input_type = tf.uint8
    # To commonalize postprocess, output_type is float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    with open(output_path, 'wb') as w:
        w.write(tflite_model)
    print('Quantization Completed!', output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--pb_path', default="./tflite/model_float32.pb")
    parser.add_argument('--output_path', default='./tflite/model_quantized.tflite')
    parser.add_argument('--calib_num', type=int, default=100, help='number of images for calibration.')
    parser.add_argument('--tfds_root', default='/mnt/d/Dataset/coco2017/')
    parser.add_argument('--download_tfds', action='store_true', help='download tfds. it takes a lot of time.')
    args = parser.parse_args()
    quantize_model(args.input_size, args.pb_path, args.output_path, args.calib_num, args.tfds_root, args.download_tfds)


