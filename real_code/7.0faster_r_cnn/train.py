import os
import cv2
import random
import tensorflow as tf
import numpy as np
from utils import compute_iou, load_gt_boxes, wandhG, compute_regression
from rpn import RPNplus
from PIL import Image
pos_thresh = 0.5
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 512, 512
def encode_label(gt_boxes):#return target
    target_scores = np.zeros(shape=[32, 32, 9, 2]) # 0: background, 1: foreground, ,
    target_bboxes = np.zeros(shape=[32, 32, 9, 4]) # t_x, t_y, t_w, t_h
    target_masks  = np.zeros(shape=[32, 32, 9]) # negative_samples: -1, positive_samples: 1
    for i in range(32): # y: height
        for j in range(32): # x: width
            for k in range(9):
                center_x = j * grid_width + grid_width * 0.5
                center_y = i * grid_height + grid_height * 0.5
                xmin = center_x - wandhG[k][0] * 0.5
                ymin = center_y - wandhG[k][1] * 0.5
                xmax = center_x + wandhG[k][0] * 0.5
                ymax = center_y + wandhG[k][1] * 0.5
                # print(xmin, ymin, xmax, ymax)
                # ignore cross-boundary anchors
                if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                    anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                    anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                    # print(anchor_boxes)
                    # compute iou between this anchor and all ground-truth boxes in image.
                    ious = compute_iou(np.array(anchor_boxes), np.array(gt_boxes))
                    # print(ious)
                    positive_masks = ious >= pos_thresh
                    negative_masks = ious <= neg_thresh

                    if np.any(positive_masks):
                        target_scores[i, j, k, 1] = 1.
                        target_masks[i, j, k] = 1 # labeled as a positive sample
                        # find out which ground-truth box matches this anchor
                        max_iou_idx = np.argmax(ious)
                        selected_gt_boxes = gt_boxes[max_iou_idx]
                        target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                    if np.all(negative_masks):
                        target_scores[i, j, k, 0] = 1.
                        target_masks[i, j, k] = -1 # labeled as a negative sample
    return target_scores, target_bboxes, target_masks

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def process_image_label(image_path, label_path):#
    raw_image = cv_imread(image_path)
    x,y,depth = raw_image.shape
    img2 = cv2.resize(raw_image,(512,512))
    gt_boxes = load_gt_boxes(label_path)
    for i in range(len(gt_boxes)):
        gt_boxes[i][0] = gt_boxes[i][0] * 512 / y
        gt_boxes[i][1] = gt_boxes[i][1] * 512 / x
        gt_boxes[i][2] = gt_boxes[i][2] * 512 / y
        gt_boxes[i][3] = gt_boxes[i][3] * 512 / x
    print(gt_boxes)
    target = encode_label(gt_boxes)
    return img2/255.0, target

def create_image_label_path_generator():
    image_paths = os.listdir('E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
    label_paths = os.listdir('E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations')

    while True:
        for i in range(len(image_paths)):
            yield os.path.join('E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',image_paths[i]),os.path.join('E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations',label_paths[i])

def DataGenerator(batch_size):
    """
    generate image and mask at the same time
    """
    image_label_path_generator = create_image_label_path_generator()
    while True:
        images = np.zeros(shape=[batch_size, image_height, image_width, 3], dtype=np.float)
        target_scores = np.zeros(shape=[batch_size, 32, 32, 9, 2], dtype=np.float)
        target_bboxes = np.zeros(shape=[batch_size, 32, 32, 9, 4], dtype=np.float)
        target_masks  = np.zeros(shape=[batch_size, 32, 32, 9], dtype=np.int)

        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, target = process_image_label(image_path, label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]
        yield images, target_scores, target_bboxes, target_masks

def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    """
    target_scores shape: [1, 45, 60, 9, 2],  pred_scores shape: [1, 45, 60, 9, 2]
    target_bboxes shape: [1, 45, 60, 9, 4],  pred_bboxes shape: [1, 45, 60, 9, 4]
    target_masks  shape: [1, 45, 60, 9]
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
    foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
    score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1,2,3]) / np.sum(foreground_background_mask)
    score_loss = tf.reduce_mean(score_loss)

    boxes_loss = tf.abs(target_bboxes - pred_bboxes)
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss<1, tf.float32) + (boxes_loss - 0.5) * tf.cast(boxes_loss >=1, tf.float32)
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
    foreground_mask = (target_masks > 0).astype(np.float32)
    boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)

    return score_loss, boxes_loss

EPOCHS = 10
STEPS = 2000
batch_size = 2
lambda_scale = 1.
TrainSet = DataGenerator(batch_size)

model = RPNplus()
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
writer = tf.summary.create_file_writer("./log")
global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

for epoch in range(EPOCHS):
    for step in range(STEPS):
        global_steps.assign_add(1)
        image_data, target_scores, target_bboxes, target_masks = next(TrainSet)
        with tf.GradientTape() as tape:
            pred_scores, pred_bboxes = model(image_data)
            score_loss, boxes_loss = compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes)
            total_loss = score_loss + lambda_scale * boxes_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f" %(epoch+1, step+1,
                                                        total_loss.numpy(), score_loss.numpy(), boxes_loss.numpy()))
        # writing summary data
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=global_steps)
            tf.summary.scalar("score_loss", score_loss, step=global_steps)
            tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
        writer.flush()
    model.save_weights("RPN.h5")

