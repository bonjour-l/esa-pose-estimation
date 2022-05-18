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
    mask = np.array(mask).astype(np.uint8)
    # mask_seg=np.asarray(mask==(3),np.int32)
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


class ESADataSet(Dataset):
    def __init__(self, root, train=True, scale=256,gauss_size=2):


        self.root = root
        self.data = []
        self.train=train
        self.scale = scale
        self.gauss_size=gauss_size
        self.img_w=1920
        self.img_h=1200
        self.img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449],
                                 std=[0.229])
        ])
        self.test_img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449],
                                 std=[0.229])
        ])

        if (self.train == True):
            with open('data/train.pkl', 'rb') as fo:  # 读取pkl文件数据
                self.train_data = pickle.load(fo, encoding='bytes')

            self.data += self.train_data


        else:
            with open('data/test.pkl', 'rb') as fo:  # 读取pkl文件数据
                self.test_data = pickle.load(fo, encoding='bytes')

            self.data += self.test_data

    def __getitem__(self, item):

        des = self.data[item]
        # print(des['K'])
        img = read_mask_np(self.root + des['rgb_pth'])


        x, y, w, h = des['bbox']

        c0=int((x+w)/2)
        c1=int((y+h)/2)
        size=int(max((w-x),(h-y))/2)

        x_new = int(c0-1.05*size)
        y_new = int(c1-1.05*size)
        w_new = int(c0+1.05*size)
        h_new = int(c1+1.05*size)
        if (w_new-x_new)!=(h_new-y_new):
            h_new=y_new+(w_new-x_new)

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

        if xsize!=size or ysize!=size:

            image=np.pad(image,((0,size-xsize),(0,size-ysize)),'edge')

        rate = 1.0
        if size != self.scale:
            rate = self.scale / size
            try:
                image = cv2.resize(image, (self.scale, self.scale))


            except Exception as e:
                print(e)

        #plt.imshow(image)
        #plt.show()


        heatmaps = []
        weights = []


        kpoints=des['sift']
        kpoints=rate*(kpoints-[x_new,y_new])

        for i in range(len(kpoints)):
            point=kpoints[i]
            heatmap = CenterLabelHeatMap(self.scale, self.scale, point[0],point[1] ,self.gauss_size)
            weight = heatmap.copy()

            weight = generate_weight_map(weight)

            heatmaps.append(heatmap)
            weights.append(weight)
        label = cv2.merge(heatmaps)

        weights = cv2.merge(weights)


        label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        weights = torch.tensor(weights, dtype=torch.float32).permute(2, 0, 1)
        start=time.clock()
        if self.train:
            image = self.img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))
            pass
        else:
            image = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))

        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # plt.imshow(mask)
        # plt.show()

        return [image,  label, weights], [bbox, rate,  des['sift'], des['sift3d'], des['K'],
                                               des['RT'].astype('float32'),des['qua'],img]

    def __len__(self):
        return len(self.data)





if __name__ == "__main__":
    train_data = ESADataSet(
        root='/home/zhaobotong/下载/speed/images/train/',
        train=True,

        scale=256,
        gauss_size=2)
    begin=time.clock()
    #data=train_data.__getitem__(40)
    print(time.clock()-begin)
    [c_data, data] = train_data.__getitem__(2964)
    image,  label, weights = c_data
    bbox, rate, sift, sift_3d, K,RT=data

    print(rate)
#    bbox,K,RT   = data
    sift=np.squeeze(sift)
    [x,y,w,h]=bbox
    sift=(sift-(x,y))*rate
    for i in range(11):
        cv2.circle(image,(int(sift[i][0]),int(sift[i][1])),3,[255],3)
    #sift=sift.reshape([15,2])

    #image=image.cpu().numpy()
    #image=image.transpose(1,2,0)

    plt.imshow(image)
    plt.show()
