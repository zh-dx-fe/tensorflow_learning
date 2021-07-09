import pandas as pd
import numpy as np
import cv2
from kmeans import *

def load_gt_boxes(path):
    gt_bboxes = []
    classes = []
    df = pd.read_csv(path,header=None)
    for i in range(len(df)):
        labels = []
        bboxes = []
        false = False
        true = True
        d = eval(df.iloc[i,5])
        for j in d['items']:
            labels.append(j['labels']['标签'])
            bboxes.append(j['meta']['geometry'])
        gt_bboxes.append(bboxes)
        classes.append(labels)

    # tree = ET.parse(path)
    # root = tree.getroot()
    # root_iter = root.iter('object')
    # for i in root_iter:
    #     x_y_max_min = []
    #     for j in list(i.getchildren()[4].getchildren()):
    #         x_y_max_min.append(float(j.text))  # xmin,ymin,xmax,ymax
    #     gt_bboxes.append(x_y_max_min)
    #     classes.append(i[0].text)
    return gt_bboxes , classes

def image_preporcess(image, target_size, gt_boxes=[]):

    ih, iw    = target_size
    h,  w, _  = image.shape
    print(h,w)
    scale = min(iw/w, ih/h)
    print(scale)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if len(gt_boxes) == 0:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def read_path(path):
    d = pd.read_csv(path,header=None)
    annotations = np.array(d.iloc[:,4])
    # np.random.shuffle(annotations)
    return annotations
def parse_annotations(i):
    img_path = img_path_o + '/' + samples[i]
    gt= gt_bboxes[i]
    img = cv_imread(img_path)
    print(gt)
    try:
        img, gt1 = image_preporcess(np.copy(img), [416,416],np.copy(gt))
    except:
        gt1 = []
    return gt1

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img


csv_path = 'E:/chrome下载/2021-7-5/2train_rname.csv'
img_path_o = 'E:/chrome下载/2021-7-5/2_images.tar'
samples = read_path(csv_path)
gt_bboxes, classes = load_gt_boxes(csv_path)
bboxes = []
for i in range(2940):
    gt_bbox_ = parse_annotations(i)
    if len(gt_bbox_) != 0:
        if len(bboxes) == 0:
            bboxes = gt_bbox_
        else:
            bboxes = np.concatenate((bboxes,gt_bbox_),0)
    print(i)
bboxes = np.array(bboxes)
anchor_bboxes = bboxes[:,[2,3]] - bboxes[:,[0,1]]
anchors = AnchorKmeans(9,max_iter=416)
anchors.fit(anchor_bboxes)