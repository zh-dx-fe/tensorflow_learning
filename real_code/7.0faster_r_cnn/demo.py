import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import compute_iou, plot_boxes_on_image, wandhG, load_gt_boxes, compute_regression, decode_output
from train import cv_imread,process_image_label


pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = 16  # 网格的长宽都是16，因为从原始图片到 feature map 经历了16倍的缩放
grid_height = 16
image_height = 720
image_width = 960
image_path = "./synthetic_dataset/synthetic_dataset/image/2.jpg"
label_path = "./synthetic_dataset/synthetic_dataset/imageAno/2.txt"
gt_boxes = load_gt_boxes(label_path)  # 把 ground truth boxes 的坐标读取出来
raw_image = cv2.imread(image_path)  # 将图片读取出来 (高，宽，通道数)
