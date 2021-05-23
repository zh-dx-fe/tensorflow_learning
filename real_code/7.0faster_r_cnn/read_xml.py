# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:59:30 2021

@author: lenovo
"""
import xml.etree.ElementTree as ET
def read_xml_x_y(path):
    res = []
    tree = ET.parse(path)
    root = tree.getroot()
    root_iter = root.iter('object')
    for i in root_iter:
        x_y_max_min = []
        for j in list(i.getchildren()[4].getchildren()):
            x_y_max_min.append(float(j.text))#xmin,ymin,xmax,ymax
        res.append(x_y_max_min)
    return res#[-1,4]

# print(read_xml_x_y('E:/chrome下载/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000039.xml'))
            
        