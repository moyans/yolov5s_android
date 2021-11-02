import tensorflow as tf
# Convert the model
import numpy as np
import os
import cv2

tf_model_path = './tflite' # openvino 输出saved_model的目录
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.float16]
# converter.allow_custom_ops = False
# converter.experimental_new_converter = True
tflite_model = converter.convert()

# Save the model
tflite_model_path = './tflite/model_float16.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)