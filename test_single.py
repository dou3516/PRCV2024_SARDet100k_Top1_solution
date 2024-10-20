import os
from os.path import join, exists
import cv2
import json
import numpy as np
from ultralytics import YOLOv10
from ensemble_boxes import *
import sys


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def infer(pt, augment=False):

    model = YOLOv10(pt)
    model.val(mode='predict', source='datasets/SARDet2024/test_A', data='SARDet_A.yaml', split='test', batch=1,
              imgsz=1024, conf=0.001, iou=0.7, max_det=100, save_txt=True, save_conf=True, augment=augment)



def txt2submit(img_dir, pred_dir, submit_json):
    files = [i for i in os.listdir(img_dir)]
    files.sort()

    dets = []
    for f in files:
        fi = join(img_dir, f)
        fl = join(pred_dir, f[:-4] + '.txt')
        img = cv2.imread(fi)
        h, w = img.shape[:2]
        with open(fl, 'rt') as fid:
            lines = fid.readlines()
            fid.close()
        labels = []
        bboxes = []
        scores = []
        for l in lines:
            ts = l.strip('\n').split(' ')
            labels.append(int(ts[0]))
            cx = float(ts[1])
            cy = float(ts[2])
            cw = float(ts[3])
            ch = float(ts[4])
            # bbox = [(cx-cw/2)*w, (cy-ch/2)*h, (cx+cw/2)*w, (cy+ch/2)*h]
            bbox = [round((cx - cw / 2) * w, 3), round((cy - ch / 2) * h, 3), round((cx + cw / 2) * w, 3), round((cy + ch / 2) * h, 3)]
            bboxes.append(bbox)
            scores.append(float(ts[5]))
        #
        det = {
            'ori_shape': (h, w),
            'img_name': f,
            'pred_instances': {
                'labels': labels,
                'bboxes': bboxes,
                'scores': scores
            }
        }
        dets.append(det)
    # save
    with open(submit_json, 'w') as fid:
        json.dump(dets, fid)  # , default=default_dump


if __name__ == '__main__':
    pt = sys.argv[1]

    img_dir = 'datasets/SARDet2024/test_A'  #
    pred_dir = 'runs/detect/val'
    os.system('rm -r %s' % pred_dir)
    submit_json = 'runs/submit_single.json'

    # infer
    print('load: %s' % pt)
    infer(pt)

    # txt2submit
    txt2submit(img_dir, join(pred_dir, 'labels'), submit_json)
    print('Finish infer and save result to %s' % submit_json)
