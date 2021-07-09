from util import *
from PIL import Image
import pandas as pd
import numpy as np
#画图
import matplotlib.pyplot as plt
#载入图片
import matplotlib.image as mpimg

df = pd.read_csv('E:/chrome下载/2021-7-5/2train_rname.csv',header=None)
bboxes,classes = load_gt_boxes('E:/chrome下载/2021-7-5/2train_rname.csv')

for i in range(8):
    false = False
    true = True
    d = eval(df.iloc[i, 5])
    bbox,_class = bboxes[i],classes[i]
    img = cv_imread('E:/chrome下载/2021-7-5/2_images.tar/'+df.iloc[i,4])
    print(df.iloc[i,4])
    print(bbox)
    # plt.imshow(img)
    img = plot_boxes_on_image(img, (bbox,_class))
    Image.fromarray(img).show()

    # h,  w, _  = img.shape
    # for j in d['items']:
    #     bbox = np.array(j['meta']['geometry'])
    #     if np.any(bbox[[0,2]]>h) or np.any(bbox[[1,3]]>w):
    #         print(bbox,h,w)
    #         df = df.drop([i,])
    # df.to_csv('E:/chrome下载/2021-7-5/2train_rname2.csv',index=False)





