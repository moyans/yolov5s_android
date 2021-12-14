from PIL import Image
import cv2
import argparse
from runner import TfLiteRunner
from utils import plot_and_save 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="bus.jpg")
    parser.add_argument('-m', '--model_path', default="/workspace/yolov5/tflite/model_float32.tflite")
    parser.add_argument('-i', '--input_size', type=int, default=640)
    parser.add_argument('--output_path', default='result.jpg')
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--quantize_mode', action='store_true')
    args = parser.parse_args()

    runner = TfLiteRunner(args.model_path, args.input_size, args.conf_thres, args.iou_thres, args.quantize_mode)
    bboxres = runner.detect(args.image)
    img_cv = cv2.imread(args.image)
    plot_and_save(bboxres, img_cv, args.output_path)

# python detect.py -m ..\yolov5_shufflenet_x0.5_stem_wd_adamw_x320\tflite_quantize\model_fp16.tflite -i 320 --image bus.jpg 
# python detect.py -m ..\yolov5_shufflenet_x0.5_stem_wd_adamw_x320\tflite_quantize\model_int8.tflite -i 320 --image bus.jpg  --quantize_mode 