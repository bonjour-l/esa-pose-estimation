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

def mask_to_bbox(mask,i=255):
    if(mask.shape.__len__()==2):
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array(i)))
    elif(mask.shape[2]==1):
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array(i)))
    else:
        mask = ma.getmaskarray(ma.masked_equal(mask, np.array([i, i, i])))[:, :, 0]
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    return [x, y, w+x, h+y]


def read_rgb_np(rgb_path):
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img,np.uint8)
    return img


def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask = np.array(mask).astype(np.int32)
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

class LinemodDataSet(Dataset):
    def __init__(self,root,train=True,use_fuse=True,use_render=True,scale=128):

        self.root=root
        self.data=[]
        self.train=train
        self.scale=scale
        self.img_transforms=transforms.Compose([
            transforms.ColorJitter(0.1,0.1,0.05,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if(self.train==True):
            if(use_fuse):
                with open("data/cat_fuse.pkl", 'rb') as fo:     # 读取pkl文件数据
                    self.fuse_data = pickle.load(fo, encoding='bytes')
            if(use_render):
                with open("data/cat_render.pkl", 'rb') as fo:     # 读取pkl文件数据
                    self.render_data = pickle.load(fo, encoding='bytes')
            with open("data/cat_real.pkl", 'rb') as fo:     # 读取pkl文件数据
                self.real_data = pickle.load(fo, encoding='bytes')
            #self.data+=self.real_data
            self.data+=self.fuse_data
            self.data+=self.render_data
        else:
            with open("data/cat_real.pkl", 'rb') as fo:     # 读取pkl文件数据
                self.real_data = pickle.load(fo, encoding='bytes')
            self.data+=self.real_data


    def __getitem__(self, item):

        des=self.data[item]
        #print(des['K'])
        img=read_rgb_np(self.root+des['rgb_pth'])
        mask=read_mask_np(self.root+des['dpt_pth'])


        if des['rgb_pth'][0]=='f':
            mask=np.asarray(mask==(3),np.uint8)
        elif len(mask.shape)==3:
            mask=np.sum(mask,2)>0
            mask=np.asarray(mask,np.uint8)
        else:
            mask=np.asarray(mask,np.uint8)




        x,y,w,h=mask_to_bbox(mask,1)

        x_new=max(0,int(1.25*x-0.25*w))
        y_new=max(0,int(1.25*y-0.25*h))
        w_new=min(640,int(1.25*w-0.25*x))
        h_new=min(480,int(1.25*h-0.25*y))
        bbox=[x_new,y_new,w_new,h_new]
        image=img[y_new:h_new,x_new:w_new]
        mask=mask[y_new:h_new,x_new:w_new]
        down=h_new-y_new
        left=w_new-x_new
        rate=1.0
        if down>self.scale or left>self.scale:
            rate=self.scale/(max(down,left))
            left=int(np.floor(left*rate))
            down=int(np.floor(down*rate))
            image=cv2.resize(image,(left,down))
            mask=cv2.resize(mask,(left,down))
        [r,g,b]=cv2.split(image)
        r=np.pad(r,((0,self.scale-down),(0,self.scale-left)),'constant')
        g=np.pad(g,((0,self.scale-down),(0,self.scale-left)),'constant')
        b=np.pad(b,((0,self.scale-down),(0,self.scale-left)),'constant')
        image=cv2.merge([r,g,b])

        #plt.imshow(image)
        #plt.show()

        mask=np.pad(mask,((0,self.scale-down),(0,self.scale-left)),'constant')


        heatmaps=[]
        weights=[]

        #for point in des['sift']:
        #    img=cv2.circle(image,(int(rate*(point[0]-x_new)),int(rate*(point[1]-y_new))),1,[0,0,255],1)
        #plt.imshow(img)
        #plt.show()

        for point in des['sift']:
            heatmap=CenterLabelHeatMap(left,down,rate*(point[0]-x_new),rate*(point[1]-y_new),10)
            heatmap=np.pad(heatmap,((0,self.scale-down),(0,self.scale-left)),'constant')
            weight=heatmap.copy()

            weight=generate_weight_map(weight)

            heatmaps.append(heatmap)
            weights.append(weight)
        label=cv2.merge(heatmaps)

        weights=cv2.merge(weights)


        if len(mask.shape)==3:
            mask=np.sum(mask,2)>0
            mask=np.asarray(mask,np.int32)
        mask=torch.tensor(np.ascontiguousarray(mask),dtype=torch.int64)
        label=torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        weights=torch.tensor(weights, dtype=torch.float32).permute(2, 0, 1)

        if self.train:
            image=self.img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))
        else:
            image=self.test_img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))


        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        #plt.imshow(mask)
        #plt.show()

        return [image,mask,label,weights],[bbox,rate,des['sift'],des['sift_3d'],des['K'],des['RT'].astype('float32')]
    def __len__(self):
        return len(self.data)

class LinemodOcclusionDataSet(Dataset):
    def __init__(self,root,scale=128):

        self.root=root
        self.data=[]
        self.scale=scale
        self.img_transforms=transforms.Compose([
            transforms.ColorJitter(0.1,0.1,0.05,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


        with open("data/occ_cat_real.pkl", 'rb') as fo:     # 读取pkl文件数据
            self.real_data = pickle.load(fo, encoding='bytes')
        self.data+=self.real_data


    def __getitem__(self, item):

        des=self.data[item]
        #print(des['K'])
        img=read_rgb_np(self.root+des['rgb_pth'])
        mask=read_mask_np(self.root+des['dpt_pth'])


        if des['rgb_pth'][0]=='f':
            mask=np.asarray(mask==(3),np.uint8)
        elif len(mask.shape)==3:
            mask=np.sum(mask,2)>0
            mask=np.asarray(mask,np.uint8)
        else:
            mask=np.asarray(mask,np.uint8)




        x,y,w,h=mask_to_bbox(mask,1)

        x_new=max(0,int(1.25*x-0.25*w))
        y_new=max(0,int(1.25*y-0.25*h))
        w_new=min(640,int(1.25*w-0.25*x))
        h_new=min(480,int(1.25*h-0.25*y))
        bbox=[x_new,y_new,w_new,h_new]
        image=img[y_new:h_new,x_new:w_new]
        mask=mask[y_new:h_new,x_new:w_new]
        down=h_new-y_new
        left=w_new-x_new
        rate=1.0
        if down>self.scale or left>self.scale:
            rate=self.scale/(max(down,left))
            left=int(np.floor(left*rate))
            down=int(np.floor(down*rate))
            image=cv2.resize(image,(left,down))
            mask=cv2.resize(mask,(left,down))
        [r,g,b]=cv2.split(image)
        r=np.pad(r,((0,self.scale-down),(0,self.scale-left)),'constant')
        g=np.pad(g,((0,self.scale-down),(0,self.scale-left)),'constant')
        b=np.pad(b,((0,self.scale-down),(0,self.scale-left)),'constant')
        image=cv2.merge([r,g,b])

        #plt.imshow(image)
        #plt.show()

        mask=np.pad(mask,((0,self.scale-down),(0,self.scale-left)),'constant')


        heatmaps=[]
        weights=[]
        #for point in des['sift']:
        #    img=cv2.circle(image,(int(rate*(point[0]-x_new)),int(rate*(point[1]-y_new))),1,[0,0,255],1)
        #plt.imshow(img)
        #plt.show()

        for point in des['sift']:
            heatmap=CenterLabelHeatMap(left,down,rate*(point[0]-x_new),rate*(point[1]-y_new),10)
            heatmap=np.pad(heatmap,((0,self.scale-down),(0,self.scale-left)),'constant')
            weight=heatmap.copy()

            weight=generate_weight_map(weight)

            heatmaps.append(heatmap)
            weights.append(weight)
        label=cv2.merge(heatmaps)

        weights=cv2.merge(weights)


        if len(mask.shape)==3:
            mask=np.sum(mask,2)>0
            mask=np.asarray(mask,np.int32)
        mask=torch.tensor(np.ascontiguousarray(mask),dtype=torch.int64)
        label=torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        weights=torch.tensor(weights, dtype=torch.float32).permute(2, 0, 1)


        image=self.test_img_transforms(Image.fromarray(np.ascontiguousarray(image, np.uint8)))


        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        #plt.imshow(mask)
        #plt.show()

        return [image,mask,label,weights],[bbox,rate,des['sift'],des['sift_3d'],des['K'],des['RT'].astype('float32')]
    def __len__(self):
        return len(self.data)
if __name__=="__main__":

    occ_test_data=LinemodOcclusionDataSet("/home/lin/Documents/6D/pvnet/pvnet-master/data/OCCLUSION_LINEMOD/")
    data0=occ_test_data.__getitem__(0)
    with open("data/occ_cat_real.pkl", 'rb') as fo:     # 读取pkl文件数据
        fuse_data = pickle.load(fo, encoding='bytes')

    print("123")