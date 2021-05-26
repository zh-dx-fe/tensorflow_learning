import numpy as np
import tensorflow as tf
from util import *
import config as cfg
import os
import random



class Data_hat(object):
    def __init__(self,dataset_type):
        self.input_size = 416
        self.batch_size = 4 if dataset_type == 'train' else 2
        self.data_aug = True if dataset_type == 'train' else False

        self.train_input_size = 416
        self.strides = np.array([16, 32])
        self.num_classes = 2
        self.anchors = [[[10,14],[23,27],[37,58]],[[81,82],[135,169],[344,319]]]/np.array([[[16]],[[32]]])
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 150

        self.gt_path_o = 'E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/Annotations/'
        self.img_path_o = 'E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/JPEGImages/'
        self.txt_path = 'E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/ImageSets/Main/'+dataset_type+'.txt'

        self.samples = read_path(self.txt_path)
        self.num_samples = len(self.samples)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def parse_annotations(self,i):
        gt_path = self.gt_path_o + self.samples[i] + '.xml'
        img_path = self.img_path_o + self.samples[i] + '.jpg'
        gt_bboxes,gt_classes = load_gt_boxes(gt_path)
        img = cv_imread(img_path)
        img, gt_bboxes = image_preporcess(np.copy(img), [self.train_input_size, self.train_input_size],
                                                   np.copy(gt_bboxes))
        gt = gt_bboxes,gt_classes
        return img,gt

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)
            # -1,416,416,3

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)


            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    image , gt = self.parse_annotations(index)
                    bboxes , classes = gt
                    bboxes = np.array(bboxes,dtype=np.float)

                    label_sbbox, label_mbbox, sbboxes, mbboxes = self.preprocess_true_boxes(
                        bboxes,classes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes

                return batch_image, (batch_smaller_target, batch_medium_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    # def parse_annotation(self, annotation):
    #
    #     line = annotation.split()
    #     image_path = line[0]
    #     if not os.path.exists(image_path):
    #         raise KeyError("%s does not exist ... " % image_path)
    #     image = cv2.imread(image_path)
    #     bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
    #
    #     if self.data_aug:
    #         image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
    #         image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
    #         image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
    #
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image, bboxes = image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
    #                                            np.copy(bboxes))
    #     return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes,classes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(2)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(2)]
        bbox_count = np.zeros((2,))

        for i in range(len(bboxes)):
            bbox_coor = bboxes[i]
            bbox_class_ind = 1 if classes[i] == 'hat' else 0

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]  # 3,4

            iou = []
            exist_positive = False
            for i in range(2):  # 8,16,32
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # center
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 3，
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox = label
        sbboxes, mbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, sbboxes, mbboxes

    def __len__(self):
        return self.num_batchs




