import sys
import torch
sys.path.insert(0, 'D:/code/TFLITE/yolov5_v5.0')
from models.experimental import attempt_load

model = attempt_load('./best.pt', map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(m.anchor_grid)