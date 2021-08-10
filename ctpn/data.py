import os
import pandas as pd
from PIL import Image,ImageDraw
import numpy as np
from util import *
from math import ceil, floor
import tensorflow as tf

class img_data(object):
    def __init__(self,data_type):
        self.batch_size = 1
        self.anchors =[6, 12, 24, 36]
        self.train_csv_path = ['E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train_20210524.csv','E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train1_20210526.csv','E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train2_20210526.csv']

        self.samples, self.dics = self.read_csv(self.train_csv_path)
        self.num_samples = len(self.samples)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.texts,self.gts = self.load_gt_boxes(self.dics)
        self.i = np.arange(0,self.num_samples)
        np.random.shuffle(self.i)

    def read_csv(self,paths):
        img_paths = []
        dics = []
        for path in paths:
            df = pd.read_csv(path)
            paths1 = df.iloc[:,1]
            for i in range(len(paths1)):
                img_paths.append(eval(paths1[i])['tfspath'])
            dic = df.iloc[:,2]
            for i in range(len(dic)):
                dic[i] = eval(dic[i])
            dic = list(dic)
            dics = dics + dic
        return img_paths, dics

    def load_gt_boxes(self,dics):
        txts = []
        gts = []
        for dic in dics:
            txt = []
            gt = []
            for j in dic[0]:
                txt.append(eval(j['text'])['text'])
                ass = j['coord']
                x_min, x_max = min(eval(ass[0]), eval(ass[2]), eval(ass[4]), eval(ass[6])), max(eval(ass[0]),
                                                                                                eval(ass[2]),
                                                                                                eval(ass[4]),
                                                                                                eval(ass[6]))
                y_min, y_max = min(eval(ass[1]), eval(ass[3]), eval(ass[5]), eval(ass[7])), max(eval(ass[1]),
                                                                                                eval(ass[3]),
                                                                                                eval(ass[5]),
                                                                                                eval(ass[7]))

                gt.append([x_min,y_min,x_max,y_max])
            txts.append(txt)
            gts.append(gt)
        return txts,gts

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            if self.batch_count < self.num_batchs:
                try:
                    no = self.i[self.batch_count]
                    img, [height_feat, width_feat], target_cls, target_ver, target_hor = \
                        get_image_and_targets(self.samples[no],self.texts[no],np.array(self.gts[no]),self.anchors,self.dics[no][1]['option'])
                    self.batch_count += 1
                    return img, [height_feat, width_feat], target_cls, target_ver, target_hor
                except:
                    self.batch_count += 1
                    return 'error!'

                # no = self.i[self.batch_count]
                # img, [height_feat, width_feat], target_cls, target_ver, target_hor = \
                #     get_image_and_targets(self.samples[no], self.texts[no], np.array(self.gts[no]), self.anchors,
                #                           self.dics[no][1]['option'])
                # self.batch_count += 1
                # return img, [height_feat, width_feat], target_cls, target_ver, target_hor

            else:
                self.batch_count = 0
                np.random.shuffle(self.i)
                raise StopIteration
    def __len__(self):
        return self.num_batchs

class crnn_data():
    def __init__(self,data_type = 'train'):
        self.batch_size = 1
        self.train_csv_path = ['E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train_20210524.csv','E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train1_20210526.csv','E:/chrome下载/2021-7-5/ocr_/Xeon1OCR_round1_train2_20210526.csv']

        self.samples, self.dics = self.read_csv(self.train_csv_path)
        self.num_samples = len(self.samples)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.texts,self.gts = self.load_gt_boxes(self.dics)
        self.i = np.arange(0,self.num_samples)
        np.random.shuffle(self.i)
        self.small_count = 0
        self.tokenizer = self.texts_to_tokenizer()

    def texts_to_tokenizer(self):
        list = []
        for text in self.texts:
            b = []
            for words in text:
                a = []
                for word in words:
                    a.append(word)
                b.append(a)
            list = list + b
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=512,filters='')
        lang_tokenizer.fit_on_texts(list)
        return lang_tokenizer



    def read_csv(self,paths):
        img_paths = []
        dics = []
        for path in paths:
            df = pd.read_csv(path)
            paths1 = df.iloc[:,1]
            for i in range(len(paths1)):
                img_paths.append(eval(paths1[i])['tfspath'])
            dic = df.iloc[:,2]
            for i in range(len(dic)):
                dic[i] = eval(dic[i])
            dic = list(dic)
            dics = dics + dic
        return img_paths, dics

    def load_gt_boxes(self,dics):
        txts = []
        gts = []
        for dic in dics:
            txt = []
            gt = []
            for j in dic[0]:
                txt.append(eval(j['text'])['text'])
                ass = j['coord']
                x_min, x_max = min(eval(ass[0]), eval(ass[2]), eval(ass[4]), eval(ass[6])), max(eval(ass[0]),
                                                                                                eval(ass[2]),
                                                                                                eval(ass[4]),
                                                                                                eval(ass[6]))
                y_min, y_max = min(eval(ass[1]), eval(ass[3]), eval(ass[5]), eval(ass[7])), max(eval(ass[1]),
                                                                                                eval(ass[3]),
                                                                                                eval(ass[5]),
                                                                                                eval(ass[7]))

                gt.append([x_min,y_min,x_max,y_max])
            txts.append(txt)
            gts.append(gt)
        return txts,gts

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            if self.batch_count < self.num_batchs:
                try:
                    no = self.i[self.batch_count]
                    len_one = len(self.gts[no])
                    gts = self.gts[no]
                    txts = self.texts[no]
                    img_path = self.samples[no]
                    if self.small_count < len_one:
                        j = self.small_count
                        if self.small_count == len_one-1:
                            self.batch_count += 1
                            self.small_count = 0
                        else:
                            self.small_count += 1
                        sequence = self.tokenizer.texts_to_sequences(txts[j])
                        sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence,padding='post')
                        return sequence, get_img_through_gt(img_path,gts[j],self.dics[no][1]['option'])

                except:
                    return 'error!'



            else:
                self.batch_count = 0
                np.random.shuffle(self.i)
                raise StopIteration


















