import time

import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.ma as ma
import os
import cv2
from loss import generate_weight_map


def mask_to_bbox(mask, i=255):
    if mask.shape.__len__() == 2:
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array(i)))
    elif mask.shape[2] == 1:
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array(i)))
    else:
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array([i, i, i])))[:, :, 0]
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w + x, h + y]


def read_rgb_np(rgb_path):
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    return img


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    #mask = np.array(mask).astype(np.uint8)
    #mask_seg=np.asarray(mask==(3),np.int32)
    return mask


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


class ESAValDataSet(Dataset):
    def __init__(self, root, real=True, scale=256,gauss_size=2):


        self.root = root
        self.data = []
        self.real=real
        self.scale = scale
        self.gauss_size=gauss_size
        self.img_w=1920
        self.img_h=1200
        self.img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                 std=[0.229])
        ])
        self.test_img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                 std=[0.229])
        ])

        if (self.real == True):
            with open('data/real_val.pkl', 'rb') as fo:  # 读取pkl文件数据
                self.train_data = pickle.load(fo, encoding='bytes')

            self.data += self.train_data


        else:
            with open('data/val.pkl', 'rb') as fo:  # 读取pkl文件数据
                self.test_data = pickle.load(fo, encoding='bytes')

            self.data += self.test_data

    def __getitem__(self, item):

        des = self.data[item]
        # print(des['K'])
        #print(des['rgb_pth'])
        img_name=des['rgb_pth']
        if self.real:
            mask = read_mask_np(self.root + 'real_test/'+des['rgb_pth'])
            img=mask.convert('L')
            img = np.array(img).astype(np.uint8)

        else:
            mask = read_mask_np(self.root + 'test/'+des['rgb_pth'])
            img=mask.convert('L')
            img = np.array(img).astype(np.uint8)







        x, y, w, h = des['bbox']

        k=1.05
        c0=int((x+w)/2)
        c1=int((y+h)/2)
        size=int(max((w-x),(h-y))/2)

        x_new = int(c0-k*size)
        y_new = int(c1-k*size)
        w_new = int(c0+k*size)
        h_new = int(c1+k*size)
        #if (w_new-x_new)!=(h_new-y_new):
            #h_new=y_new+(w_new-x_new)

        if x_new<0:
            w_new-=x_new
            x_new=0
        if y_new<0:
            h_new-=y_new
            y_new=0
        if w_new>self.img_w:

            x_new=x_new+self.img_w-w_new
            if x_new<0:
                x_new=0
            w_new=self.img_w

        if h_new>self.img_h:
            y_new=y_new+self.img_h-h_new
            if y_new<0:
                y_new=0
            h_new=self.img_h
        bbox = [x_new, y_new, w_new, h_new]

        size = max(w_new-x_new,h_new - y_new)

        xsize=w_new-x_new
        ysize=h_new-y_new
        image = img[y_new:h_new, x_new:w_new]
        #cv2.imshow('img',image)
        #cv2.waitKey(0)
        if xsize!=size or ysize!=size:

            image=np.pad(image,((0,size-xsize),(0,size-ysize)),'edge')

        rate = 1.0
        if size != self.scale:
            rate = self.scale / size
            try:
                image = cv2.resize(image, (self.scale, self.scale))


            except Exception as e:
                print(e)

        #image2=cv2.medianBlur(image,3)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        #dst = cv2.filter2D(image, -1, kernel)

        #cv2.imshow("img",dst)
        #cv2.waitKey(0)
        #image=dst

        image = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))



        return [image,bbox, rate,   des['sift3d'], des['K'],img_name,img]

    def __len__(self):
        return len(self.data)





if __name__ == "__main__":
    with open('data/real_val.pkl', 'rb') as fo:  # 读取pkl文件数据
        test_data = pickle.load(fo, encoding='bytes')

    train_data = ESAValDataSet(
        root='/home/zhaobotong/下载/speed/images/',
        real=True,
        scale=128,
        gauss_size=2)
    begin=time.clock()
    #data=train_data.__getitem__(40)
    print(time.clock()-begin)
    for i in range(train_data.__len__()):
        data = train_data.__getitem__(i)
        image, bbox, rate,   sift3d, K,img_name,_=data


        print(img_name)

        #cv2.imshow("img",image)
        #cv2.waitKey(0)
