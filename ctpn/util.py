import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

from math import ceil, floor

'''
#
dir_data = './data_train'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''








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
    #
    # no text
    if maxIoU <= 0:  #
        #
        num_anchors = len(anchor_heights)

        # (k, 2)
        cls = [0, 1] * num_anchors
        ver = [0, 0] * num_anchors
        hor = [0, 0] * num_anchors
        #
        return cls, ver, hor
    #
    # text
    cls = []
    ver = []
    hor = []
    #
    for idx, ah in enumerate(anchor_heights):
        #
        if not idx == anchor_posi:
            cls.extend([0, 1])  #
            ver.extend([0, 1])
            hor.extend([0, 1])
            continue
        #
        cls.extend([1, 0])  #
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
        ver.extend([ratio_bbox[1], ratio_bbox[3]])
        hor.extend([ratio_bbox[0], ratio_bbox[2]])
        #
    # list_len = 2k
    # cls表示2k个prob,ver和hor分别表示竖直方向和水平方向的回归率
    return cls, ver, hor
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
        img = img.rotate(270, expand=True)
    elif orientation == '底部朝上':
        img = img.rotate(180, expand=True)
    elif orientation == '底部朝右':
        img = img.rotate(90, expand=True)
    h1 , w1 = img.size
    img = img.resize((408,int(ceil(w1*(408/h1)))))

    img_data = np.array(img, dtype=np.float32) / 255
    # height, width, channel
    #
    img_data = img_data[:, :, 0:3]  # rgba
    #

    # texts
    #

    # targets
    img_size = img_data.shape  # height, width, channel
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
    return [img_data], [height_feat, width_feat], target_cls, target_ver, target_hor
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


#
if __name__ == '__main__':
    #
    print('draw target bbox ... ')
    #
    import model_detect_meta as meta

    #
    list_imgs = get_files_with_ext(meta.dir_images_valid, 'png')
    #
    curr = 0
    NumImages = len(list_imgs)
    #
    # valid_result save-path
    if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
    #
    for img_file in list_imgs:
        #
        txt_file = get_target_txt_file(img_file)
        #
        img_data, feat_size, target_cls, target_ver, target_hor = \
            get_image_and_targets(img_file, txt_file, meta.anchor_heights)
        #
        curr += 1
        print('curr: %d / %d' % (curr, NumImages))
        #
        filename = os.path.basename(img_file)
        arr_str = os.path.splitext(filename)
        #
        # image
        '''
        r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
        g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
        b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
        #
        file_target = os.path.join(meta.dir_results_valid, 'target_' +arr_str[0] + '.png')
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        '''

        file_target = os.path.join(meta.dir_results_valid, 'target_' + arr_str[0] + '.png')
        img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
        img_target.save(file_target)

        #
        # trans
        text_bbox, conf_bbox = trans_results(target_cls, target_ver, target_hor, \
                                             meta.anchor_heights, meta.threshold)
        #
        draw_text_boxes(file_target, text_bbox)
        #
    #
    print('draw end.')
    #

