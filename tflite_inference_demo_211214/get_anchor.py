# -*- coding: utf-8 -*-
# Created on 12月-14-21 14:34
# @site: https://github.com/moyans
# @author: moyan
import sys
import torch
import argparse
sys.path.insert(0, 'D:/code/TFLITE/yolov5_v5.0')
from models.experimental import attempt_load

def main(model_path):
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    # print(m.anchor_grid)
    
    print(m.anchor_grid[0].shape)
    for i in range(3):
        print('for {} stage:'.format(i))
        stg_list = []
        for j in range(3):
            x1 = m.anchor_grid[i][0][j][0][0][0].numpy().tolist()
            x2 = m.anchor_grid[i][0][j][0][0][1].numpy().tolist()
            # print(type(x1))
            stg_list.append(x1)
            stg_list.append(x2)
        print(stg_list)
    print(len(m.anchor_grid))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', default="bus.jpg")
    args = parser.parse_args()

    main(args.model_path)

    # python get_anchor.py ..\yolov5_shufflenet_x0.5_stem_wd_adamw_x320\best.pt