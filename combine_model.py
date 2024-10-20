import torch
from collections import  OrderedDict
from ultralytics import YOLOv10

split_num = 10
pt1 = 'runs/sardet_x_1024_e100_b48_a2_ft3/train/weights/epoch38.pt'
pt2 = 'runs/sardet_xstarv1_1024_e100_b48_ft2/train/weights/epoch35.pt'

ckpt1 = torch.load(pt1, map_location="cpu")
csd1 = ckpt1['model'].float().state_dict()
ckpt2 = torch.load(pt2, map_location="cpu")
csd2 = ckpt2['model'].float().state_dict()

model = YOLOv10('ultralytics/cfg/models/diys/yolov10x_starv2.yaml')
csd_keep = OrderedDict()
for key in csd1.keys():
    if int(key.split('.')[1]) > split_num:
        csd_keep[key] = csd1[key]
for key in csd2.keys():
    if int(key.split('.')[1]) <= split_num:
        csd_keep[key] = csd2[key]
model.model.load_state_dict(csd_keep)
# print(len(csd_keep.keys()))
torch.save(model, 'yolov10x_starv2.pt')

model = YOLOv10('ultralytics/cfg/models/diys/yolov10x_starv3.yaml')
csd_keep = OrderedDict()
for key in csd1.keys():
    if int(key.split('.')[1]) <= split_num:
        csd_keep[key] = csd1[key]
for key in csd2.keys():
    if int(key.split('.')[1]) > split_num:
        csd_keep[key] = csd2[key]
model.model.load_state_dict(csd_keep)
# print(len(csd_keep.keys()))
torch.save(model, 'yolov10x_starv3.pt')
