import os
from os.path import join, exists
from tqdm import tqdm
import numpy as np


def combine_trainval():
    base_path = 'datasets/SARDet2024'
    train = join(base_path, 'train')
    val = join(base_path, 'val')
    trainval = join(base_path, 'trainval')
    trainval_images = join(trainval, 'images')
    trainval_labels = join(trainval, 'labels')
    for cdir in [trainval, trainval_images, trainval_labels]:
        if not exists(cdir):
            os.makedirs(cdir)

    for sub in ['train', 'val']:
        sub_images = join(base_path, sub, 'images')
        files = [i for i in os.listdir(sub_images)]
        for f in files:
            fi = join(sub_images, f)
            fo = join(trainval_images, f)
            cmd = 'ln -s %s %s' % (fi, fo)
            os.system(cmd)
        sub_labels = join(base_path, sub, 'labels')
        files = [i for i in os.listdir(sub_labels)]
        for f in files:
            fi = join(sub_labels, f)
            fo = join(trainval_labels, f)
            cmd = 'ln -s %s %s' % (fi, fo)
            os.system(cmd)


combine_trainval()