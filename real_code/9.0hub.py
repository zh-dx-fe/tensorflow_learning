import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
image_path = 'E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
image_tensor = tf.expand_dims(tf.constant(cv_imread(image_path),dtype=tf.uint8),axis=0)
print(image_tensor.shape)

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
detector_output = detector(image_tensor)
class_ids = detector_output["detection_classes"]