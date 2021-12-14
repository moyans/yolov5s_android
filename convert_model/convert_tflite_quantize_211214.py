# -*- coding: utf-8 -*-
# Created on 12月-14-21 10:06
# @site: https://github.com/moyans
# @author: moyan
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
'''
request:
    tensorflow==2.6.0
'''

def convert_fp16_quantize(pb_path, save_path):

    input_arrays = ['inputs']
    output_arrays = ['Identity', 'Identity_1', 'Identity_2']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.float16]
    # converter.allow_custom_ops = False
    # converter.experimental_new_converter = True
    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)

def convert_int8_quantize(pb_path, save_path, calDir, calNum, input_size):
    
    def load_from_cv_only_resize(img_path, input_size):
        img = cv2.imread(img_path)
        im = cv2.resize(img, [input_size, input_size])
        im = im[::-1] # bgr->rgb
        im = np.ascontiguousarray(im)
        return im

    def representative_dataset_gen():
        img_list = os.listdir(calDir)
        use_calib_num =  len(img_list)  if len(img_list) < calNum else calNum
        for i in range(use_calib_num):
            print('calibrating...', i)
            img_name = img_list[i]
            img_path = os.path.join(calDir, img_name)
            assert os.path.exists(img_path)
            im = load_from_cv_only_resize(img_path, input_size)
            im = im.astype(np.float32) / 255.
            if len(im.shape) == 3: im = im[None]
            print('  input tensor shape: ', im.shape)
            yield [im]
    
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

    with open(save_path, 'wb') as w:
        w.write(tflite_model)
    print('Quantization Completed!', save_path)

def test_int8():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--pb_path', default="./yolov5_shufflenet_x0.5_stem_wd_adamw_x320/tflite/model_float32.pb")
    parser.add_argument('--output_path', default='./yolov5_shufflenet_x0.5_stem_wd_adamw_x320/tflite_quantize/model_int8.tflite')
    parser.add_argument('--calib_num', type=int, default=100, help='number of images for calibration.')
    parser.add_argument('--calib_dir', default='./test_img',)

    args = parser.parse_args()
    convert_int8_quantize(args.pb_path, args.output_path, calDir=args.calib_dir, calNum=args.calib_num, input_size=args.input_size)

def test_fp16():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path', default="./yolov5_shufflenet_x0.5_stem_wd_adamw_x320/tflite/model_float32.pb")
    parser.add_argument('--output_path', default='./yolov5_shufflenet_x0.5_stem_wd_adamw_x320/tflite_quantize/model_fp16.tflite')
    args = parser.parse_args()
    convert_fp16_quantize(args.pb_path, args.output_path)

if __name__ == '__main__':
    # test_int8()
    test_fp16()
    
