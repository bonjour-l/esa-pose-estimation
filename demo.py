import os

import cv2
import torch
import heapq
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from data_load4 import ESADataSet
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
from lib.utils.base_utils import Projector
from visual import visualize_bounding_box


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
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.mseloss = nn.MSELoss(reduce=False)
        self.hwing_loss = Loss_weighted()


    def forward(self, image, heatmaps, weights,  train=True):
        start = time.clock()
        heatmaps_pred = self.net(image)
        # print(time.clock()-start)

        if not train:
            return heatmaps_pred, 0

        loss_vertex = self.hwing_loss(heatmaps_pred, heatmaps, weights)


        return heatmaps_pred,  loss_vertex


def val(net,  index):
    net.eval()
    class_name='esa'
    scale=256

    train_data =  ESADataSet(
        root='/home/zhaobotong/下载/speed/images/train/',
        train=True,

        scale=128,
        gauss_size=2)
    test_data =  ESADataSet(
        root='/home/zhaobotong/下载/speed/images/train/',
        train=False,

        scale=128,
        gauss_size=2)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    losses_ver = AverageMeter()
    losses_seg = AverageMeter()
    losses = AverageMeter()

    dis_gauss = AverageMeter()
    dis_max = AverageMeter()

    run_time = AverageMeter()
    loss_low = AverageMeter()
    # loss_middle=0
    loss_high = AverageMeter()
    loss_total = AverageMeter()
    losses_ver.reset()
    losses_seg.reset()
    losses.reset()

    loss_total.reset()
    dis_gauss.reset()
    dis_max.reset()

    # load_model(net.module.net, optimizer, './net_'+class_name, index*10-1)
    # load_model(net.module.net, optimizer, './net_cat_hwing_0.2', index*10-1)


    # print(eval.linemod_db.get_diameter('cat'))
    print("eval ", index, " epoch")
    trans = []
    degrees =[]
    score_tran=[]
    score_degree=[]
    pixerror = [[] for _ in range(32)]
    idx=0
    dis_rt=[]
    results=[]
    for i, data in enumerate(test_data_loader, 0):
        # get the inputs
        # c_data=data[0]
        # data=data[1]

        c_data, data = data
        image, heatmaps, weights = [d.cuda() for d in c_data]
        bbox, rate, farthest, farthest_3d, K, RT,qua,img = data

        begin = time.clock()

        heatmaps_pred, loss_vertex = net(image,  heatmaps, weights)
        #print(loss_tran.detach().cpu().numpy())
        # print(stop-begin)
        '''
        for i in range(heatmaps.shape[0]):
            hp = heatmaps[i]
            hpp = heatmaps_pred[i]
            low_ind = hp < 0.1
            high_ind = hp > 0.1
            loss_low.update(torch.mean(torch.abs(hp[low_ind] - hpp[low_ind])).detach(), n=torch.sum(low_ind))
            # loss_middle+=torch.mean(torch.abs(hp[middle_ind]-hpp[middle_ind]))
            loss_high.update(torch.mean(torch.abs(hp[high_ind] - hpp[high_ind])).detach(), n=torch.sum(high_ind))
            loss_total.update(torch.mean(torch.abs(hp - hpp)).detach(), n=128 * 128 * 32)
        '''
        # print(loss_low)
        # print(loss_high)

        '''
        preds,maxvals =getPrediction(heatmaps_pred)
        preds=np.squeeze(preds.cpu().detach().numpy())
        maxvals=np.squeeze(maxvals.cpu().detach().numpy())
        '''


        x, y, w, h = [np.squeeze(d.numpy()) for d in bbox]

        # print(pred_pose)

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
            # co.append([b[0][i][d[0][i]-1].cpu().numpy().item(),d[0][i]])

        # print(maxvals)
        heatmaps_pred = heatmaps_pred.cpu().detach().numpy()

        preds = get_final(heatmaps_pred, co)
        farthest = np.squeeze(farthest.detach().cpu().numpy())
        #targets=(farthest-(x,y))*rate.numpy()

        large_k=np.sum(np.asarray(maxvals)>0.6)
        #print(large_k)

        #large_k=max(large_k,24)

        idxs = heapq.nlargest(large_k, range(len(maxvals)), maxvals.__getitem__)
        # print(idxs)

        '''
        #heatmaps_pred=heatmaps_pred.cpu().detach().numpy()
        heatmaps=heatmaps.cpu().detach().numpy()

        for i in range(11):

            if not i in idxs:
                continue

            print(maxvals[i])

            img=image[0]
            img=img.permute(1,2,0)
            img=img.detach().cpu().numpy()
            img=np.squeeze(img)


            hp=heatmaps_pred[0,i]
            label=heatmaps[0,i]
            weight=weights[0,i].detach().cpu().numpy()
            

            hp=np.abs(hp)
            hp=hp/np.max(hp)
            t=np.zeros(hp.shape,dtype=np.float32)
            hp=np.abs(hp-label)
            print(np.linalg.norm(hp,ord=1))
            t=cv2.merge([hp,img,label])
            #p=np.abs(hp).argmax()
            p=preds[i]
            print(p)
            #cv2.circle(t,(int(p[0]),int(p[1])),1,[0,0,255],1)

            p=farthest[i]
            p=(p-(x,y))*rate.numpy()
            print(p)
            #cv2.circle(t,(int(p[0]),int(p[1])),1,[0,255,0],1)


            plt.imshow(t)
            plt.show()
        '''

        ori_preds = preds * (1 / rate.numpy()) + [x, y]
        ori_preds_m = preds_max * (1 / rate.numpy()) + (x, y)

        # print(ori_preds)
        # print(ori_preds_m)

        points_3d = np.squeeze(farthest_3d.detach().cpu().numpy())
        '''***************************'''
        points_2d = ori_preds

        p3d = points_3d[idxs]
        #p2d=farthest[idxs]
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
        t[0]=t[0]
        t[1]=t[1]
        r_exp=camera[:3]
        r,_=cv2.Rodrigues(r_exp)

        pose_pred = np.concatenate((r, np.asarray(t.reshape(3, 1))), axis=1)
        pose_targets=np.squeeze(RT.detach().cpu().numpy())
        img=np.squeeze(img.permute(1,2,0).cpu().numpy())
        img=np.stack((img,)*3,axis=-1)
        projector = Projector()
        bb8_3d = np.loadtxt('./data/esa_bb8_3d.txt')
        bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'esa')
        bb8_2d_gt = projector.project(bb8_3d, pose_targets, 'esa')
        #plt.imshow(img)
        #plt.show()
        #visualize_bounding_box(img, bb8_2d_pred[None, None, ...], bb8_2d_gt[None, None, ...],save=True,save_fn='./savetest/'+str(idx)+'.png')




        # print(pose_pred)
        stop = time.clock()
        # print(stop-begin)
        run_time.update(stop - begin)

        pred_t=pose_pred[:, 3]
        target_t=pose_targets[:, 3]
        score_t=np.linalg.norm(pred_t-target_t,ord=2)/np.linalg.norm(target_t,ord=2)
        score_tran.append(score_t)

        target_qua=np.squeeze(qua.cpu().numpy())
        r3 = R.from_matrix(pose_pred[:, :3])
        q3 = r3.as_quat()
        pred_qua=np.asarray([q3[3],q3[0],q3[1],q3[2]],dtype=np.float32)

        res=np.hstack([pred_t,pred_qua])
        #print(res)
        results.append(res)
        score_r=2*np.real(np.arccos(np.abs(np.matmul(pred_qua,target_qua))+0j))
        #print(np.abs(np.matmul(pred_qua,target_qua)),score_r)
        score_degree.append(score_r)

        dist=pose_pred[:, 3] - pose_targets[:, 3]
        translation_distance = np.abs(pose_pred[:, 3] - pose_targets[:, 3])
        trans.append(translation_distance)

        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        drt=np.asarray([dist[0],dist[1],dist[2],angular_distance])
        if not np.isnan(angular_distance):
            degrees.append(angular_distance)
            dis_rt.append(drt)
        print(len(trans),np.mean(score_tran, axis=0),np.mean(score_degree, axis=0))


        # fc=open('cat_pix.txt','a')
        for ii in range(len(idxs)):
            i = idxs[ii]
            # for i in range(32):
            p_f = ori_preds[i]
            p_m = ori_preds_m[i]
            f = farthest[i]
            # print(p_f,p_m,f)
            epix = np.linalg.norm(f - p_f)
            dis_gauss.update(np.linalg.norm(f - p_f))
            dis_max.update(np.linalg.norm(f - p_m))
            # pixerror[i].append(epix)
            # fc.write(str(epix)+'\n')



        idx=idx+1
    mean_tran = np.mean(trans, axis=0)

    #print(np.mean(trans, axis=0))
    '''
    with open('add_dis/'+class_name+'.txt','w') as fc:
        for dis in eval.add_dists:
            fc.write(str(dis)+'\n')
    '''
    #np.savetxt('dix_rt_3.txt',dis_rt)
    #np.savetxt('results_3.txt',results)
    print(run_time.sum, run_time.count, run_time.avg)

    print('Gaiss', dis_gauss.avg)

    ret=[class_name,str(index),str(round(np.mean(score_tran, axis=0),5)),str(round(np.mean(score_degree, axis=0),5)),str(round(dis_gauss.avg,5)),str(round(np.mean(trans),5)),str(round(np.mean(degrees),5))]
    fi = open('load/load_' + class_name + '.txt', 'a')
    fi.write(ret[0] + '\t' + ret[1] + '\t' + ret[2] + '\t'
             + ret[3] + '\t'+ ret[4] + '\t'
             + ret[5] + '\t'+ ret[6] + '\n')
    fi.close()
    print(ret[2]+'\t'+ret[3]+'\n')

    # plt.imshow(image[0].cpu().permute(1,2,0))
    # plt.show()

    return ret


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
    #class_names = ['ape', 'benchvise','cam','can', 'cat', 'duck', 'driller', 'eggbox', 'glue','holepuncher', 'iron', 'lamp', 'phone']
    #class_names=['ape','can','cat','duck','driller','eggbox','glue','holepuncher']
    # net=Resnet50_8s(ver_dim=32)
    class_names=['esa']
    #class_names=['ape', 'benchvise','cam','can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'iron', 'lamp', 'phone']
    #class_names=[ 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp','phone']
    net = seg_hrnet3.get_seg_model(config)
    net = NetWrapper(net)
    net = DataParallel(net).cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    for class_name in class_names:

        idx='best_rotate'
        load_model(net.module.net, optimizer, './net_' + class_name, idx)
        # for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        ret=val(net,  index=idx)



    # load_model(net.module.net, optimizer, './net_'+class_name, 9)
    # val(net,class_name,9)
    print("123")
