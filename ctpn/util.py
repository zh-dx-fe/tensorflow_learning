import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

from math import ceil, floor
import math
import cv2
import copy

'''
#
dir_data = './data_train'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''



def get_small_gts(gts):
    sm_b_l = []
    for gt in gts:
        sm_bs = []
        xmin, ymin, xmax, ymax = gt
        grid_left = int(xmin//8)
        grid_right = int(xmax//8)
        sm_bs.append([xmin,ymin,(grid_left+1)*8,ymax])
        for m in range(grid_left+1,grid_right):
            sm_bs.append([m*8,ymin,(m+1)*8,ymax])
        sm_bs.append([grid_right*8,ymin,xmax,ymax])
        sm_b_l.extend(sm_bs)
    return sm_b_l



def bbox_overlaps(boxes,query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=float)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

# def anchor_target(gts, anchors):
#     _anchors = np.vstack([np.full([len(anchors)], 8), np.array(anchors), np.full([len(anchors)], 8), np.array(anchors)]).transpose()
#     _num_anchors = _anchors.shape[0]
#     height, width = 32,51
#     # 1. Generate proposals from bbox deltas and shifted anchors
#     shift_x = np.arange(0, width) * 8  # (51,) (0, 408, 8)
#     shift_y = np.arange(2, height+2) * 12  # (32,) (24, 408, 12)
#     shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order # (51, 32)
#     # K is H x W
#     shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
#                         shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量 (H*W, 4)
#     A = _num_anchors  # 4个anchor
#     K = shifts.shape[0]  # feature-map的宽乘高的大小 # H*W
#     all_anchors = (_anchors.reshape((1, A, 4)) +
#                    shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加 # (K, A, 4)
#     all_anchors = all_anchors.reshape((K * A, 4))  # (K*A, 4)
#     total_anchors = int(K * A)


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = np.array(image).shape
    if _ == 4:
        image = np.delete(image, -1, axis=-1)
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = image.resize((nw,nh))
    # image_resized = cv2.resize(image, (nw, nh))
    image_resized = np.array(image_resized)

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes






#
# 对得到的txt_list每一个词计算anchor并挑出满足条件的
# 获得cls, ver, hor,每个格子最多只有一个文本 哪怕每个格子都有k个anchor_heights 而且取中心点最近的进行预测
def calculate_targets_at(anchor_center, txt_list, gts, anchor_heights):
    #
    # anchor_center = [hc, wc]
    #
    #
    # anchor width:  8,
    # anchor height: 6, 12, 24, 36, ...
    #
    # anchor stride: 12,8
    #

    #
    anchor_width = 8
    #
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width
    #

    #
    hc = anchor_center[0]
    wc = anchor_center[1]
    #
    maxIoU = 0
    anchor_posi = 0
    text_bbox = []
    #
    for item in zip(gts,txt_list):
        #
        # width: if more than half of the anchor is text, positive;
        # height: if more than half of the anchor is text, positive;
        # heigth_IoU: of the 4 anchors, choose the one with max height_IoU;
        #
        bbox = item[0]
        #
        # horizontal
        flag = 0
        #
        if (bbox[0] < wc and wc <= bbox[2]):
            flag = 1
        elif (wc < bbox[0] and bbox[2] < wc + asw):
            if (bbox[0] - wc < wc + asw - bbox[2]):
                flag = 1
        elif (wc - asw < bbox[0] and bbox[2] < wc):
            if (bbox[2] - wc >= wc - asw - bbox[0]):
                flag = 1
        #
        if flag == 0: continue
        #
        # vertical
        #
        bcenter = (bbox[1] + bbox[3]) / 2.0
        #
        d0 = abs(hc - bcenter)
        dm = abs(hc - ash - bcenter)
        dp = abs(hc + ash - bcenter)
        #
        if (d0 < ash and d0 <= dm and d0 < dp):
            pass
        else:
            continue
            #
        #
        posi = 0
        #
        for ah in anchor_heights:
            #
            hah = ah // 2  # half_ah
            #
            IoU = 1.0 * (min(hc + hah, bbox[3]) - max(hc - hah, bbox[1])) \
                  / (max(hc + hah, bbox[3]) - min(hc - hah, bbox[1]))
            #
            if IoU > maxIoU:
                maxIoU = IoU
                anchor_posi = posi
                text_bbox = bbox
            #
            posi += 1
            #
        #
        break



    cls = np.zeros([len(anchor_heights),2])  # (A, 2)
    ver = np.zeros([len(anchor_heights),2])
    hor = np.zeros([len(anchor_heights),2])
    # 计算所有负样本
    anchors_ = np.array([[wc-4, hc-ahh/2, wc+4, hc+ahh/2] for ahh in anchor_heights])
    gts_small = np.array(get_small_gts(gts))
    overlaps = bbox_overlaps(anchors_.astype(float), gts_small.astype(float))  # (A, G)
    argmax_overlaps = overlaps.argmax(axis=1)  # (A,) #找到和每一个gtbox，overlap最大的那个anchor
    max_overlaps = overlaps[np.arange(len(anchor_heights)), argmax_overlaps]  # (A,)
    cls[max_overlaps < 0.3,:] = np.array([0,1])  # 标注负样本
    for i in range(len(anchors_)):
        if anchors_[i][1] < 0 or anchors_[i][3] >= 408:
            cls[i,:] = 0

    # no text
    if maxIoU <= 0:  #
        #
        #
        return cls.reshape((-1)), ver.reshape((-1)), hor.reshape((-1))
    #
    # text

    #
    for idx, ah in enumerate(anchor_heights):
        #
        if not idx == anchor_posi:

            continue
        #
        cls[idx,:] = np.array([1, 0])  #标注正样本
        #
        half_ah = ah // 2
        half_aw = anchor_width // 2
        #
        anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
        #
        ratio_bbox = [0, 0, 0, 0]
        #
        ratio = (text_bbox[0] - anchor_bbox[0]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[0] = ratio
        #
        # print(ratio)
        #
        ratio = (text_bbox[2] - anchor_bbox[2]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[2] = ratio
        #
        # print(ratio)
        #
        ratio_bbox[1] = (text_bbox[1] - anchor_bbox[1]) / ah
        ratio_bbox[3] = (text_bbox[3] - anchor_bbox[3]) / ah
        #
        # print(ratio_bbox)
        #
        ver[idx,:] = np.array([ratio_bbox[1], ratio_bbox[3]])
        hor[idx,:] = np.array([ratio_bbox[0], ratio_bbox[2]])
        #
    # list_len = 2k
    # cls表示2k个prob,ver和hor分别表示竖直方向和水平方向的回归率
    return cls.reshape((-1)), ver.reshape((-1)), hor.reshape((-1))
    #


#
# util function
# 得到文本框、目标框、三个分支的anchor
def get_image_and_targets(img_file, txt_list, gts, anchor_heights, orientation):
    # img_data
    response = requests.get(img_file)
    response = response.content

    BytesIOObj = BytesIO()
    BytesIOObj.write(response)
    img = Image.open(BytesIOObj)
    if orientation == '底部朝左':
        img = img.rotate(90, expand=True)
    elif orientation == '底部朝上':
        img = img.rotate(180, expand=True)
    elif orientation == '底部朝右':
        img = img.rotate(270, expand=True)
    # w1, h1 = img.size
    # d_h = 408/h1
    # img = img.resize((int(ceil(w1*d_h)),408))
    # gts = np.ceil(np.array(gts) * d_h).astype(int)
    #
    # img_data = np.array(img, dtype=np.float32) / 255
    # # height, width, channel
    # #
    img_data,gts = image_preporcess(img,[408,408],np.array(gts))  # (408, 408, 3), (G, 4)
    img_data = img_data[:, :, 0:3]  # rgba


    # texts
    #

    # targets
    img_size = img_data.shape  # height, width, channel
    img_data = np.expand_dims(img_data,0)
    #
    # ///2, ///2, //3, -2
    # ///2, ///2, ///2,
    #
    height_feat = floor(ceil(ceil(img_size[0] / 2.0) / 2.0) / 3.0) - 2
    width_feat = ceil(ceil(ceil(img_size[1] / 2.0) / 2.0) / 2.0)
    #

    #
    num_anchors = len(anchor_heights)
    #
    target_cls = np.zeros((height_feat, width_feat, 2 * num_anchors))
    target_ver = np.zeros((height_feat, width_feat, 2 * num_anchors))
    target_hor = np.zeros((height_feat, width_feat, 2 * num_anchors))
    #

    #
    # detection
    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8],
    #
    # anchor width:  8,
    # anchor height: 12, 24, 36, 48,
    #
    # feature_layer --> receptive_field
    # [0,0] --> [0:36, 0:8]
    # [0,1] --> [0:36, 8:8+8]
    # [i,j] --> [12*i:36+12*i, 8*j:8+8*j]
    #
    # feature_layer --> anchor_center
    # [0,0] --> [18, 4]
    # [0,1] --> [18, 4+8]
    # [i,j] --> [18+12*i, 4+8*j]
    #

    #
    # anchor_width = 8
    #
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width
    #
    hc_start = 18  # 6*2 + 6
    wc_start = 4  # 4
    #

    for h in range(height_feat):
        #
        hc = hc_start + ash * h  # anchor height center
        #
        for w in range(width_feat):
            #
            cls, ver, hor = calculate_targets_at([hc, wc_start + asw * w], txt_list, gts, anchor_heights)
            # (h, w, 2k) * 3
            target_cls[h, w] = cls
            target_ver[h, w] = ver
            target_hor[h, w] = hor
            #
    #
    return img_data, [height_feat, width_feat], target_cls, target_ver, target_hor
    #


# 对每一个预测来的特征图 遍历每一个格子 若没有一个anchor达到threshold 则直接跳过
# 若有达到的 则取最大的那个 同时获取ver和hor计算预测的textbbox
def trans_results(r_cls, r_ver, r_hor, anchor_heights, threshold):
    #
    # anchor width: 8,
    #

    #
    anchor_width = 8
    #
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width
    #
    hc_start = 18
    wc_start = 4
    #

    #
    aw = anchor_width
    #

    #
    list_bbox = []
    list_conf = []
    #
    feat_shape = r_cls.shape
    # print(feat_shape)
    #
    for h in range(feat_shape[0]):
        #
        for w in range(feat_shape[1]):
            #
            if max(r_cls[h, w, :]) < threshold: continue
            #
            anchor_posi = np.argmax(r_cls[h, w, :])  # in r_cls
            anchor_id = anchor_posi // 2  # in anchor_heights
            #
            # print(anchor_id)
            # print(r_cls[h,w,:])
            #
            #
            ah = anchor_heights[anchor_id]  #
            anchor_posi = anchor_id * 2  # for retrieve in r_ver, r_hor
            #
            hc = hc_start + ash * h  # anchor center
            wc = wc_start + asw * w  # anchor center
            #
            half_ah = ah // 2
            half_aw = aw // 2
            #
            anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
            #
            text_bbox = [0, 0, 0, 0]
            #
            text_bbox[0] = anchor_bbox[0] + aw * r_hor[h, w, anchor_posi]
            text_bbox[1] = anchor_bbox[1] + ah * r_ver[h, w, anchor_posi]
            text_bbox[2] = anchor_bbox[2] + aw * r_hor[h, w, anchor_posi + 1]
            text_bbox[3] = anchor_bbox[3] + ah * r_ver[h, w, anchor_posi + 1]
            #
            list_bbox.append(text_bbox)
            list_conf.append(max(r_cls[h, w, :]))
            #
    # (-1, 4), (-1, 2)
    return list_bbox, list_conf
    #


# 从存在anchor的特征图的左上方格子遍历到右下方格子 若相邻格子的anchor边界小于50且overlap大于0.7 则两者合并 知道不满足合并条件 生成this_text_bbox 加入conn_bbox
def do_nms_and_connection(list_bbox, list_conf):
    max_margin = 50
    len_list_box = len(list_bbox)
    conn_bbox = []
    head = tail = 0
    for i in range(1, len_list_box):
        distance_i_j = abs(list_bbox[i][0] - list_bbox[i - 1][0])
        overlap_i_j = overlap(list_bbox[i][1], list_bbox[i][3], list_bbox[i - 1][1], list_bbox[i - 1][3])
        if distance_i_j < max_margin and overlap_i_j > 0.7:
            tail = i
            if i == len_list_box - 1:
                this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
                conn_bbox.append(this_test_box)
                head = tail = i
        else:
            this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
            conn_bbox.append(this_test_box)
            head = tail = i
    # (n, 4)
    return conn_bbox


def overlap(h_up1, h_dw1, h_up2, h_dw2):
    """
    :param h_up1:
    :param h_dw1:
    :param h_up2:
    :param h_dw2:
    :return:
    """
    overlap_value = (min(h_dw1, h_dw2) - max(h_up1, h_up2)) \
                    / (max(h_dw1, h_dw2) - min(h_up1, h_up2))
    return overlap_value


#
def draw_text_boxes(img_file, text_bbox):
    #
    # 打开图片，画图
    img_draw = Image.open(img_file)
    #
    draw = ImageDraw.Draw(img_draw)
    #
    for item in text_bbox:
        #
        xs = item[0]
        ys = item[1]
        xe = item[2]
        ye = item[3]
        #
        line_width = 1  # round(text_size/10.0)
        draw.line([(xs, ys), (xs, ye), (xe, ye), (xe, ys), (xs, ys)],
                  width=line_width, fill=(255, 0, 0))
        #
    #
    img_draw.save(img_file)
    #



