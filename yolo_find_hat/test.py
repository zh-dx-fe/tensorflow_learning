from util import *
from PIL import Image
import config as cfg
from dataset import Data_hat


# gt_path = 'E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/Annotations/'+'000002.xml'
# img_path = 'E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/JPEGImages/'+'000002.jpg'
#
# boxes = load_gt_boxes(gt_path)
# img = cv_imread(img_path)
# show_img = plot_boxes_on_image(img,boxes)
#
# Image.fromarray(show_img).show()

# print(read_path('E:/2021-5-13下载/PERSON_HAT/VOC2028/VOC2028/ImageSets/Main/train.txt'))
data = Data_hat('train')
image_data, target = next(data)

