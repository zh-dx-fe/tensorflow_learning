import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def load_gt_boxes(path):
    gt_bboxes = []
    classes = []
    tree = ET.parse(path)
    root = tree.getroot()
    root_iter = root.iter('object')
    for i in root_iter:
        x_y_max_min = []
        for j in list(i.getchildren()[4].getchildren()):
            x_y_max_min.append(float(j.text))  # xmin,ymin,xmax,ymax
        gt_bboxes.append(x_y_max_min)
        classes.append(i[0].text)
    return gt_bboxes , classes

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def plot_boxes_on_image(show_image_with_boxes, boxes, color=[0, 0, 255], thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    boxes_b = boxes[0]
    classes_ = boxes[1]
    for i in range(len(boxes_b)):
        box = boxes_b[i]
        classes = classes_[i]
        cv2.rectangle(show_image_with_boxes,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])), color=color, thickness=thickness)
        cv2.putText(show_image_with_boxes,classes,(int(box[0]),int(box[1])),font,0.25,color = (255,255,255))
    show_image_with_boxes = cv2.cvtColor(show_image_with_boxes, cv2.COLOR_BGR2RGB)
    return show_image_with_boxes

def read_path(path):
    with open(path,'r') as file:
        txt = file.readlines()
        annotations = [line.strip() for line in txt]
    np.random.shuffle(annotations)
    return annotations


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

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




