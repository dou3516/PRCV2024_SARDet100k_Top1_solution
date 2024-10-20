# PRCV2024_SARDet100k_Top1_solution (updating)
This repo is the top1 solution (team XXXR) for PRCV2024 SARDet contest.
# TODOs
- [ ] update README
- [x] add code 
# Solution
**challenges:**
1. SARDet100k is of the most Complex(large, multi-scale, diverse sources and polarization, small objects) SAR detection datasets, which takes several times the training time of a general dataset.
2. Training strategies for optical images may not always be applicable to SAR datasets.

**key ideas:**
1. Train faster:
   - use real-time network YOLOv10-xl as baseline to get faster training speed
   - Introduce starnet to get a variant model
   - Cross-combining the two previously trained models to get extral two models with good pre-trained weights
2. The right data augmentations
   - larger size
   - smaller scale variation
   - weaker hue variation
3. Multi-stage training
   - smaller learning rate
   - weaker and more augmentations
5. TTA and multi-model results integration
   - TTA (flip)
   - Weighted Boxes Fusion(WBF) of 4 models
# Setup Environment
```
pip install -r requirements.txt
pip install -e .
pip install einops ensemble_boxes timm
```
# Data preparation
```
# cd to code directory
mkdir datasets
# convert to YOLO format
python converter.py datasets/source/Annotations
cp -r datasets/source/JPEGImages/train datasets/SARDet2024/train/images
cp -r datasets/source/JPEGImages/val datasets/SARDet2024/val/images
# combine train val
python datapro_SARDet.py
# link test data to default dir
ln -s dir_of_test datasets/SARDet2024/test_A
```
# Train a model
```
# The computing resources of the organizer are 8 RTX3090
yolo detect train data=SARDet_tv.yaml model=yolov10x.pt epochs=200 batch=48 imgsz=1024 project=runs/sardet_x_1024_e200_b48 close_mosaic=0 device=0,1,2,3,4,5,6,7
```
# Test a model
```
python test_single.py runs/sardet_x_1024_e200_b48/train/weights/last.pt
```
