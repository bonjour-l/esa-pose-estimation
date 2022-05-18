import os

import cv2
import torch
import heapq
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from data_load_val import ESAValDataSet
from lib.utils.base_utils import Projector
from net import Resnet18_8s, Resnet50_8s
from torch.nn import DataParallel
from loss import Loss_weighted
from models import seg_hrnet2,seg_hrnet3
from config import config
import matplotlib.pyplot as plt
from pnp import pnp
from inference import get_final_preds, get_max_preds, getPrediction, get_final, get_final2
from evaluation import Evaluator
from evaluation import AverageMeter
from kp6d.p_poseNMS import pose_nms
import cpnp
from scipy.spatial.transform import Rotation as R
from submission import SubmissionWriter
from utils import project
from visual import visualize_bounding_box


def visualize( img,q,r, ax=None):

    """ Visualizing image, with ground truth pose with axes projected to training image. """

    if ax is None:
        ax = plt.gca()

    ax.imshow(img)



    xa, ya = project(q, r)
    ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
    ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
    ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')
    plt.show()
    return

def load_model(model, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0
    pths = []
    for pth in os.listdir(model_dir):
        if pth.split('.')[-1] == 'pth':
            pths.append(str(pth.split('.')[0]))

    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = 'best_add'
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    model.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load model {} epoch {}'.format(model_dir, pretrained_model['epoch']))
    return pretrained_model['epoch'] + 1


def visualize_mask(mask):
    plt.imshow(mask[0].cpu())
    plt.show()


class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net



    def forward(self, image):
        start = time.clock()
        heatmaps_pred = self.net(image)
        # print(time.clock()-start)




        return heatmaps_pred


def val(net,  index):
    net.eval()
    class_name='esa'
    scale=256


    test_data =  ESAValDataSet(
        root='/home/zhaobotong/下载/speed/images/',
        real=False,

        scale=128,
        gauss_size=2)
    test_real_data =  ESAValDataSet(
        root='/home/zhaobotong/下载/speed/images/',
        real=True,

        scale=128,
        gauss_size=2)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    test_real_data_loader = torch.utils.data.DataLoader(test_real_data, batch_size=1, shuffle=False, num_workers=4)





    run_time = AverageMeter()
    loss_low = AverageMeter()
    # loss_middle=0



    # load_model(net.module.net, optimizer, './net_'+class_name, index*10-1)
    # load_model(net.module.net, optimizer, './net_cat_hwing_0.2', index*10-1)


    # print(eval.linemod_db.get_diameter('cat'))
    print("eval ", index, " epoch")

    pixerror = [[] for _ in range(32)]
    submission = SubmissionWriter()
    
    idxval=0
    for i, data in enumerate(test_data_loader, 0):

        print("eval test")
        c_data = data
        image, bbox, rate,   farthest_3d, K,img_name,img=c_data
        image=image.cuda()


        begin = time.clock()

        heatmaps_pred = net(image)

        x, y, w, h = [np.squeeze(d.numpy()) for d in bbox]


        heatmaps_pred = heatmaps_pred.detach()
        a, b = torch.max(heatmaps_pred, dim=3)
        c, d = torch.max(a, dim=2)

        co = []
        preds_max = []
        maxvals = []

        for i in range(30):
            co.append(np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))
            preds_max.append(
                np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))

            maxvals.append(a[0][i][d[0][i]].cpu().numpy().item())

        heatmaps_pred = heatmaps_pred.cpu().detach().numpy()

        preds = get_final(heatmaps_pred, co)



        large_k=np.sum(np.asarray(maxvals)>0.8)
        #print(large_k)

        large_k=max(large_k,24)

        idxs = heapq.nlargest(large_k, range(len(maxvals)), maxvals.__getitem__)


        ori_preds = preds * (1 / rate.numpy()) + [x, y]


        # print(ori_preds)
        # print(ori_preds_m)

        points_3d = np.squeeze(farthest_3d.detach().cpu().numpy())

        points_2d = ori_preds

        p3d = points_3d[idxs]
        p2d = points_2d[idxs]
        mav=np.asarray(maxvals)[idxs]
        camera_K=np.squeeze(K.cpu().numpy())
        pose_pred = pnp(p3d, p2d, camera_K, cv2.SOLVEPNP_EPNP)


        r_exp,_=cv2.Rodrigues(pose_pred[:, :3])
        t=pose_pred[:, 3]
        camera=np.concatenate([np.asarray(r_exp.reshape(3)),np.asarray(t.reshape(3))],axis=0)
        #camera=cpnp.cpnp(p3d,p2d,K,camera)

        camera=cpnp.cpnp_m(p3d,p2d,mav,K,camera)
        t=camera[3:]
        #t[0]=t[0]+0.01
        #t[1]=t[1]+0.01
        r_exp=camera[:3]
        r,_=cv2.Rodrigues(r_exp)

        pose_pred = np.concatenate((r, np.asarray(t.reshape(3, 1))), axis=1)
        
        img=np.squeeze(img.permute(1,2,0).cpu().numpy())
        img=np.stack((img,)*3,axis=-1)
        projector = Projector()
        bb8_3d = np.loadtxt('./data/esa_bb8_3d.txt')
        bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'esa')
        #plt.imshow(img)
        #plt.show()
        #visualize_bounding_box(img, bb8_2d_pred[None, None, ...], corners_targets=None,save=True,save_fn='./saveval/'+str(idxval)+'.png')
        

        r3 = R.from_matrix(pose_pred[:, :3])
        q3 = r3.as_quat()
        pred_t=pose_pred[:, 3]
        pred_qua=np.asarray([q3[3],q3[0],q3[1],q3[2]],dtype=np.float32)
        img_name=img_name[0]
        

        submission.append_test(img_name, pred_qua, pred_t)

        print(pred_t,pred_qua)
        # print(pose_pred)
        stop = time.clock()
        idxval=idxval+1
        
    idxreal=0
    for i, data in enumerate(test_real_data_loader, 0):

        #print("eval real")
        c_data = data
        image, bbox, rate,   farthest_3d, K,img_name,img=c_data
        image=image.cuda()


        begin = time.clock()

        heatmaps_pred = net(image)

        x, y, w, h = [np.squeeze(d.numpy()) for d in bbox]


        heatmaps_pred = heatmaps_pred.detach()
        a, b = torch.max(heatmaps_pred, dim=3)
        c, d = torch.max(a, dim=2)

        co = []
        preds_max = []
        maxvals = []

        for i in range(30):
            co.append(np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))
            preds_max.append(
                np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))

            maxvals.append(a[0][i][d[0][i]].cpu().numpy().item())

        heatmaps_pred = heatmaps_pred.cpu().detach().numpy()

        preds = get_final(heatmaps_pred, co)



        large_k=np.sum(np.asarray(maxvals)>0.8)
        #print(large_k)

        large_k=max(large_k,24)

        idxs = heapq.nlargest(large_k, range(len(maxvals)), maxvals.__getitem__)


        ori_preds = preds * (1 / rate.numpy()) + [x, y]


        # print(ori_preds)
        # print(ori_preds_m)

        points_3d = np.squeeze(farthest_3d.detach().cpu().numpy())
        points_2d = ori_preds

        p3d = points_3d[idxs]
        p2d = points_2d[idxs]
        mav=np.asarray(maxvals)[idxs]
        camera_K=np.squeeze(K.cpu().numpy())
        pose_pred = pnp(p3d, p2d, camera_K, cv2.SOLVEPNP_EPNP)

        r_exp,_=cv2.Rodrigues(pose_pred[:, :3])
        t=pose_pred[:, 3]
        camera=np.concatenate([np.asarray(r_exp.reshape(3)),np.asarray(t.reshape(3))],axis=0)
        #camera=cpnp.cpnp(p3d,p2d,K,camera)

        camera=cpnp.cpnp_m(p3d,p2d,mav,K,camera)
        t=camera[3:]
        #t[0]=t[0]+0.01
        #t[1]=t[1]+0.01
        r_exp=camera[:3]
        r,_=cv2.Rodrigues(r_exp)

        pose_pred = np.concatenate((r, np.asarray(t.reshape(3, 1))), axis=1)

        img=np.squeeze(img.permute(1,2,0).cpu().numpy())
        img=np.stack((img,)*3,axis=-1)
        projector = Projector()
        bb8_3d = np.loadtxt('./data/esa_bb8_3d.txt')
        bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'esa')
        #bb8_2d_gt = projector.project(bb8_3d, pose_targets, 'esa')
        #plt.imshow(img)
        #plt.show()
        #visualize_bounding_box(img, bb8_2d_pred[None, None, ...], corners_targets=None,save=True,save_fn='./savevalreallast/'+str(idxreal)+'.png')

        r3 = R.from_matrix(pose_pred[:, :3])
        q3 = r3.as_quat()
        pred_t=pose_pred[:, 3]
        #if pred_t[2]>5 or pred_t[2] <3:
            #pred_t[2]=4
        pred_qua=np.asarray([q3[3],q3[0],q3[1],q3[2]],dtype=np.float32)
        img_name=img_name[0]

        #img=np.squeeze(img.permute(1,2,0).cpu().numpy())
        #visualize(img,pred_qua,pred_t)

        submission.append_real_test(img_name, pred_qua, pred_t)
        print(img_name,idxreal)
        print(pred_t,pred_qua)
        # print(pose_pred)
        stop = time.clock()
        idxreal=idxreal+1
    submission.export(suffix='no0.01_new')







'''
    for i in range(32):

        hp=heatmaps_pred[0,i]
        label=heatmaps[0,i].detach().cpu().numpy()

        hp=np.abs(hp)
        hp=hp/np.max(hp)
        t=np.zeros(hp.shape,dtype=np.float32)

        t=cv2.merge([hp,label,t])
        #p=np.abs(hp).argmax()
        p=preds[i]
        print(p)
        cv2.circle(t,(int(p[0]),int(p[1])),2,[0,0,255],2)

        plt.imshow(t)
        plt.show()

    
    preds=preds*(1/rate.numpy())+(x,y)
    farthest=np.squeeze(farthest.cpu().numpy())
    farthest_3d=np.squeeze(farthest_3d.cpu().numpy())
    K=np.squeeze(K.cpu().numpy())
    print(farthest)
    print(preds)
    print(pnp(farthest_3d,preds,K,cv2.SOLVEPNP_EPNP))
    #print(pnp(farthest_3d,farthest,K))
    print(RT)

'''

if __name__ == '__main__':

    class_names=['esa']

    net = seg_hrnet3.get_seg_model(config)
    net = NetWrapper(net)
    net = DataParallel(net).cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    for class_name in class_names:

        idx='best_rotate'
        load_model(net.module.net, optimizer, './net_' + class_name, idx)
        # for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        val(net,  index=idx)




    print("123")
