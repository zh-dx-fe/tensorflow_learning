import os
import cv2
import random
import tensorflow as tf
import numpy as np
from utils import compute_iou, load_gt_boxes, wandhG, compute_regression, plot_boxes_on_image
from rpn import RPNplus
from PIL import Image



def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def process_image_label(image_path, label_path):#
    raw_image = cv_imread(image_path)
    x,y,depth = raw_image.shape
    print(x,y)
    raw_image = cv2.resize(raw_image,(512,512))
    gt_boxes = load_gt_boxes(label_path)
    for i in range(len(gt_boxes)):
        # gt_boxes[i][0] = gt_boxes[i][0] * 511 / (x-1) - 511 / (x-1) + 1
        # gt_boxes[i][1] = gt_boxes[i][1] * 511 / (y-1) - 511 / (y-1) + 1
        # gt_boxes[i][2] = gt_boxes[i][2] * 511 / (x-1) - 511 / (x-1) + 1
        # gt_boxes[i][3] = gt_boxes[i][3] * 511 / (y-1) - 511 / (y-1) + 1
        gt_boxes[i][0] = gt_boxes[i][0] * 512 / y
        gt_boxes[i][1] = gt_boxes[i][1] * 512 / x
        gt_boxes[i][2] = gt_boxes[i][2] * 512 / y
        gt_boxes[i][3] = gt_boxes[i][3] * 512 / x
    # print(gt_boxes)
    # target = encode_label(gt_boxes)
    gt = np.array(gt_boxes,dtype=int)
    return raw_image, gt

image_path = 'E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
label_path = 'E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000009.xml'

img,gt = process_image_label(image_path,label_path)
print(gt)
Image.fromarray(plot_boxes_on_image(img,gt)).show()


target_scores = np.zeros(shape=[45, 60, 9, 2]) # 0: background, 1: foreground, ,
target_bboxes = np.zeros(shape=[45, 60, 9, 4]) # t_x, t_y, t_w, t_h
target_masks  = np.zeros(shape=[45, 60, 9]) # negative_samples: -1, positive_samples: 1

encoded_image = np.copy(img)  # 再复制原始图片
pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = 16  # 网格的长宽都是16，因为从原始图片到 feature map 经历了16倍的缩放
grid_height = 16
image_height = 512
image_width = 512
for i in range(32):
    for j in range(32):
        for k in range(9):
            center_x = j * grid_width + grid_width * 0.5  # 计算此小块的中心点横坐标
            center_y = i * grid_height + grid_height * 0.5  # 计算此小块的中心点纵坐标
            xmin = center_x - wandhG[k][0] * 0.5  # wandhG 是预测框的宽度和长度，xmin 是预测框在图上的左上角的横坐标
            ymin = center_y - wandhG[k][1] * 0.5  # ymin 是预测框在图上的左上角的纵坐标
            xmax = center_x + wandhG[k][0] * 0.5  # xmax 是预测框在图上的右下角的纵坐标
            ymax = center_y + wandhG[k][1] * 0.5  # ymax 是预测框在图上的右下角的纵坐标
            # ignore cross-boundary anchors
            if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                # compute iou between this anchor and all ground-truth boxes in image.
                ious = compute_iou(anchor_boxes, gt)
                positive_masks = ious > pos_thresh
                negative_masks = ious < neg_thresh

                if np.any(positive_masks):
                    plot_boxes_on_image(encoded_image, anchor_boxes, thickness=1)
                    print("=> Encoding positive sample: %d, %d, %d" %(i, j, k))
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[255,0,0], thickness=4)  # 正预测框的中心点用红圆表示

                    target_scores[i, j, k, 1] = 1.  # 表示检测到物体
                    target_masks[i, j, k] = 1 # labeled as a positive sample
                    # find out which ground-truth box matches this anchor
                    max_iou_idx = np.argmax(ious)
                    selected_gt_boxes = gt[max_iou_idx]
                    target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                if np.all(negative_masks):
                    target_scores[i, j, k, 0] = 1.  # 表示是背景
                    target_masks[i, j, k] = -1 # labeled as a negative sample
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[0,0,0], thickness=4)  # 负预测框的中心点用黑圆表示
Image.fromarray(encoded_image).show()
