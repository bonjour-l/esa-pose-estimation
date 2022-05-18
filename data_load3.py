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
    mask = np.array(mask).astype(np.int32)
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


class LinemodDataSet(Dataset):
    def __init__(self, root, name='cat', train=True, use_fuse=True, use_render=True, scale=128,gauss_size=2):

        self.linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone',
                                'benchvise','can','driller','eggbox','holepuncher','lamp']
        self.name=name
        self.root = root
        self.data = []
        self.train = train
        self.scale = scale
        self.gauss_size=gauss_size
        self.img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if (self.train == True):
            if use_render:
                fuse_filename = "data2/" + name + "_render.pkl"
                with open(fuse_filename, 'rb') as fo:  # 读取pkl文件数据
                    data = pickle.load(fo, encoding='bytes')
                    self.render_data=data[:10000]
            if (use_fuse):
                fuse_filename = "data2/" + name + "_fuse.pkl"
                with open(fuse_filename, 'rb') as fo:  # 读取pkl文件数据
                    self.fuse_data = pickle.load(fo, encoding='bytes')
            real_filename = "data2/" + name + "_real.pkl"
            with open(real_filename, 'rb') as fo:  # 读取pkl文件数据
                self.real_data = pickle.load(fo, encoding='bytes')
                self.real_train_data = []
                train_filename = "data2/" + name + "_train.pkl"
                with open(train_filename, 'rb') as ftrain:
                    train_split = pickle.load(ftrain, encoding='bytes')
                    for data in train_split:
                        str = data[0]
                        train_index = str.split('/')[-1].split('.')[0]
                        train_index = int(train_index)
                        train_data_i = self.real_data[train_index]
                        self.real_train_data.append(train_data_i)
                        # print(train_index)

            self.data += self.real_train_data
            if use_render:
                self.data += self.render_data
            if use_fuse:
                self.data += self.fuse_data

        else:
            real_filename = "data2/" + name + "_real.pkl"
            with open(real_filename, 'rb') as fo:  # 读取pkl文件数据
                self.real_data = pickle.load(fo, encoding='bytes')
                self.real_test_data = []
                test_filename = "data2/" + name + "_test.pkl"
                with open(test_filename, 'rb') as ftest:
                    test_split = pickle.load(ftest, encoding='bytes')
                    for data in test_split:
                        str = data[0]
                        test_index = str.split('/')[-1].split('.')[0]
                        test_index = int(test_index)
                        test_data_i = self.real_data[test_index]
                        self.real_test_data.append(test_data_i)
                        # print(train_index)

            self.data += self.real_test_data

    def __getitem__(self, item):

        des = self.data[item]
        # print(des['K'])
        img = read_rgb_np(self.root + des['rgb_pth'])
        mask = read_mask_np(self.root + des['dpt_pth'])

        if des['rgb_pth'][0] == 'f':
            #mask = np.asarray(mask == (3), np.uint8)
            mask = np.asarray(mask == (self.linemod_cls_names.index(self.name)+1), np.uint8)
        elif len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.uint8)
        else:
            mask = np.asarray(mask, np.uint8)
        x, y, w, h = des['bbox']

        c0=int((x+w)/2)
        c1=int((y+h)/2)
        size=int(max(self.scale,max((w-x),(h-y)))/2)

        x_new = int(c0-1.1*size)
        y_new = int(c1-1.1*size)
        w_new = int(c0+1.1*size)
        h_new = int(c1+1.1*size)
        if (w_new-x_new)!=(h_new-y_new):
            h_new=y_new+(w_new-x_new)

        if x_new<0:
            w_new-=x_new
            x_new=0
        if y_new<0:
            h_new-=y_new
            y_new=0
        if w_new>640:
            x_new=x_new+640-w_new
            w_new=640
        if h_new>480:
            y_new=y_new+480-h_new
            h_new=480
        bbox = [x_new, y_new, w_new, h_new]
        down = h_new - y_new
        left = w_new - x_new
        size=max(self.scale,max(left,down))
        if size>left:
            dis=size-left
            if w_new+dis<640:
                w_new+=dis
            else:
                x_new-=dis
                if x_new<0:
                    w_new-=x_new
                    x_new=0
        if size>down:
            dis=size-down
            if h_new+dis<480:
                h_new+=dis
            else:
                y_new-=dis
                if y_new<0:
                    h_new-=y_new
                    y_new=0

        image = img[y_new:h_new, x_new:w_new]

        mask = mask[y_new:h_new, x_new:w_new]

        rate = 1.0
        if size >= self.scale:
            rate = self.scale / size
            try:
                image = cv2.resize(image, (self.scale, self.scale))
                mask = cv2.resize(mask, (self.scale, self.scale))
            except Exception as e:
                print(e)

        #plt.imshow(image)
        #plt.show()


        heatmaps = []
        weights = []

        #for point in des['sift']:
        #    img=cv2.circle(image,(int(rate*(point[0]-x_new)),int(rate*(point[1]-y_new))),1,[0,0,255],1)
        #plt.imshow(img)
        #plt.show()
        kpoints=des['sift']
        #for point in des['sift']:
        for i in range(32):
            point=kpoints[i]
            heatmap = CenterLabelHeatMap(self.scale, self.scale, rate * (point[0] - x_new), rate * (point[1] - y_new),self.gauss_size)
            weight = heatmap.copy()

            weight = generate_weight_map(weight)

            heatmaps.append(heatmap)
            weights.append(weight)
        label = cv2.merge(heatmaps)

        weights = cv2.merge(weights)

        if len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.int32)
        mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
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
        return [image,  label, weights], [img,bbox, rate, des['sift'], des['sift_3d'], des['K'],
                                               des['RT'].astype('float32')]

    def __len__(self):
        return len(self.data)


class LinemodOcclusionDataSet(Dataset):
    def __init__(self, root,name,scale=128):

        self.linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone',
                                'benchvise','can','driller','eggbox','holepuncher','lamp']
        self.name=name
        self.root = root
        self.data = []
        self.scale = scale
        self.img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.gauss_size=2
        with open("data2/occ/"+name+"_real.pkl", 'rb') as fo:  # 读取pkl文件数据
            self.real_data = pickle.load(fo, encoding='bytes')
        self.data += self.real_data

    def __getitem__(self, item):

        des = self.data[item]
        # print(des['K'])
        img = read_rgb_np(self.root + des['rgb_pth'])
        mask = read_mask_np(self.root + des['dpt_pth'])

        if des['rgb_pth'][0] == 'f':
            #mask = np.asarray(mask == (3), np.uint8)
            mask = np.asarray(mask == (self.linemod_cls_names.index(self.name)+1), np.uint8)
        elif len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.uint8)
        else:
            mask = np.asarray(mask, np.uint8)




        x, y, w, h = des['bbox']

        c0=int((x+w)/2)
        c1=int((y+h)/2)
        size=int(max(self.scale,max((w-x),(h-y)))/2)

        x_new = int(c0-1.1*size)
        y_new = int(c1-1.1*size)
        w_new = int(c0+1.1*size)
        h_new = int(c1+1.1*size)
        if (w_new-x_new)!=(h_new-y_new):
            h_new=y_new+(w_new-x_new)

        if x_new<0:
            w_new-=x_new
            x_new=0
        if y_new<0:
            h_new-=y_new
            y_new=0
        if w_new>640:
            x_new=x_new+640-w_new
            w_new=640
        if h_new>480:
            y_new=y_new+480-h_new
            h_new=480
        bbox = [x_new, y_new, w_new, h_new]
        down = h_new - y_new
        left = w_new - x_new
        size=max(self.scale,max(left,down))
        if size>left:
            dis=size-left
            if w_new+dis<640:
                w_new+=dis
            else:
                x_new-=dis
                if x_new<0:
                    w_new-=x_new
                    x_new=0
        if size>down:
            dis=size-down
            if h_new+dis<480:
                h_new+=dis
            else:
                y_new-=dis
                if y_new<0:
                    h_new-=y_new
                    y_new=0

        image = img[y_new:h_new, x_new:w_new]

        mask = mask[y_new:h_new, x_new:w_new]

        rate = 1.0
        if size >= self.scale:
            rate = self.scale / size
            try:
                image = cv2.resize(image, (self.scale, self.scale))
                mask = cv2.resize(mask, (self.scale, self.scale))
            except Exception as e:
                print(e)

        #plt.imshow(image)
        #plt.show()


        heatmaps = []
        weights = []

        #for point in des['sift']:
        #   img=cv2.circle(image,(int(rate*(point[0]-x_new)),int(rate*(point[1]-y_new))),1,[0,0,255],1)
        #plt.imshow(img)
        #plt.show()
        kpoints=des['sift']
        #for point in des['sift']:
        for i in range(32):
            point=kpoints[i]
            heatmap = CenterLabelHeatMap(self.scale, self.scale, rate * (point[0] - x_new), rate * (point[1] - y_new),self.gauss_size)
            weight = heatmap.copy()

            weight = generate_weight_map(weight)

            heatmaps.append(heatmap)
            weights.append(weight)
        label = cv2.merge(heatmaps)

        weights = cv2.merge(weights)

        if len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.int32)
        mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        weights = torch.tensor(weights, dtype=torch.float32).permute(2, 0, 1)


        image = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))

        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # plt.imshow(mask)
        # plt.show()
        return [image,  label, weights], [img,bbox, rate, des['sift'], des['sift_3d'], des['K'],
                                               des['RT'].astype('float32')]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_data = LinemodDataSet(
        root='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/',
        train=True,
        use_render=True,
        use_fuse=True,
        name='cat', scale=128,
        gauss_size=2)
    begin=time.clock()
    data=train_data.__getitem__(40)
    print(time.clock()-begin)
    [c_data, data] = train_data.__getitem__(0)
    image,  label, weights = c_data
    _,bbox, rate, sift, sift_3d, K,RT=data
    #print(tran)
    #print(rate)
    print(sift)
    x,y,w,h=bbox
    #    bbox,K,RT   = data
    sift=np.squeeze(sift)
    for i in range(8):
        cv2.circle(image,(int((sift[i][0]-x)*rate),int((sift[i][1]-y)*rate)),1,[0,0,255],1)
    #sift=sift.reshape([15,2])

    #image=image.cpu().numpy()
    #image=image.transpose(1,2,0)

    plt.imshow(image)
    plt.show()
