# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import torchvision
import torch
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import cv2


root = "/home/hzq/dataset/kitti/city/seqs"
output="/home/hzq/GAN_demo/datasets/kitti"
count_gray = 0
count_mask = 0
gray_path = []
mask_path = []


def compress(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[1:375, 0:1242]
    dst_img = np.zeros([187,621], dtype=np.uint8)
    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    n = 0
    for i in range(0, rows, 2):
        m = 0
        for j in range(0,cols,2):
            dst_img[n, m] = img[i,j]
            m += 1
        n += 1
    return dst_img

for first_folder in os.listdir(root):
    gray_path_ = []
    mask_path_ = []
    for gray_imgs in os.listdir(root +'/' + first_folder + '/2011_09_26/' + first_folder + '/image_00/data'):
        gray_path_.append(root +'/' + first_folder + '/2011_09_26/' + first_folder + '/image_00/data/' + gray_imgs)
    for mask_imgs in os.listdir(root +'/' + first_folder + '/2011_09_26/' + first_folder + '/project_image'):
        mask_path_.append(root + '/' + first_folder + '/2011_09_26/' + first_folder + '/project_image/' + mask_imgs)
    gray_path_.sort()
    mask_path_.sort()
    gray_path.extend(gray_path_)
    mask_path.extend(mask_path_)


for im in gray_path:
    img = cv2.imread(im)
    img = compress(img)
    name = '%05d' % count_gray
    cv2.imwrite(output + '/trainA/' + name + '_A.jpg', img)
    count_gray = count_gray + 1
    
for im in mask_path:
    img = cv2.imread(im)
    img = compress(img)
    name = '%05d' % count_mask
    cv2.imwrite(output + '/trainB/' + name + '_B.jpg', img)
    count_mask = count_mask + 1


# im2 = cv2.imread("/home/hzq/2.png")
# im2 = cv2.resize(im2, (621, 187), interpolation=cv2.INTER_AREA)
# im3 = cv2.resize(im2, (1242, 375), interpolation=cv2.INTER_LINEAR)
# cv2.imwrite("/home/hzq/3.png", im2)
# cv2.imwrite("/home/hzq/4.png", im3)

# img = Image.open(r'/home/hzq/2.png')
# out = img.resize((621, 187), Image.LINEAR)
# out.save(r'/home/hzq/3.png')