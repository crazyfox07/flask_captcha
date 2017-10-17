# -*- coding:utf-8 -*-
"""
File Name: util
Version:
Description:
Author: liuxuewen
Date: 2017/9/20 18:06
"""
import numpy as np
import string
import os
import re
from PIL import Image
import json
# s={"0": "\u554a", "1": "\u963f", "2": "\u57c3", "3": "\u6328", "4": "\u54ce", "5": "\u5509", "6": "\u54c0", "7": "\u7691", "8": "\u764c", "9": "\u853c", "10": "\u77ee"}
# from chinese2img import common_hanzi_dict
import random

# s = common_hanzi_dict
IMAGE_HEIGHT = 152
IMAGE_WIDTH = 40
LEN_CAPTCHA = 4
# CHARS = ''.join(list(s.values()))
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

LEN_CHAR_SET = len(CHARS)

# img_path = r'D:\tmp'
img_train_path = r'/root/liuxuewen/image/train'
img_test_path = r'/root/liuxuewen/image/test'
# imgs_train = os.listdir(img_train_path)
# L = len(imgs_train)


# print(L)
# print(CHARS)

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        # print(img.shape)
        gray = np.mean(img, -1)
        return gray
    else:
        return img


# 文本转向量
def text2vec(text):
    v = np.zeros(len(CHARS) * LEN_CAPTCHA)
    for i, num in enumerate(text):
        index = i * LEN_CHAR_SET + CHARS.index(num)
        v[index] = 1
    return v


# 向量转文本
def vec2text(vec):
    text = list()
    for i, j in enumerate(vec):
        if j == 1:
            index = i % LEN_CHAR_SET
            char = CHARS[index]
            text.append(char)
    return ''.join(text)


# 生成一个训练batchv  一个批次为 默认100 张图片 转换为向量
def get_next_batch(batch_size=100, img_path=img_train_path):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, LEN_CAPTCHA * LEN_CHAR_SET])
    imgs_name = random.sample(os.listdir(img_path), batch_size)
    for i, img_name in enumerate(imgs_name):
        # 获取标签
        try:
            text = re.findall('_(\w+)\.png', img_name)[0]
            img = Image.open(r'{}\{}'.format(img_path, img_name))
        except:
            print(img_name)
            continue
            # print(text)
        # 获取图片，并灰度转换

        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # w代表宽度，h代表高度，最后一个参数指定采用的算法
        img = np.array(img)
        img = convert2gray(img)

        batch_x[i, :] = img.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def get_img(img_path):
    img = Image.open(img_path)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # w代表宽度，h代表高度，最后一个参数指定采用的算法
    img = np.array(img)
    img = convert2gray(img)
    img = img.flatten() / 255
    return np.reshape(img, (1, img.size))

if __name__ == '__main__':
    img_path=r'D:\tmp\test2\gray.png'
    img = Image.open(img_path)
    img=np.array(img)
    print(img.shape)
    print(img)
    img = convert2gray(img)
    print(img.shape)
    print(img)
    img = img.flatten() / 255
    print(img.shape)
    print(img)

