import os

import cv2
import torch
import heapq
import time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from data_load4 import LinemodDataSet, LinemodOcclusionDataSet
from net import Resnet18_8s, Resnet50_8s
from torch.nn import DataParallel
from loss import Loss_weighted
from models import seg_hrnet2
from config import config
import matplotlib.pyplot as plt
from pnp import pnp
from inference import get_final_preds, get_max_preds, getPrediction, get_final, get_final2
from evaluation import Evaluator
from evaluation import AverageMeter
from kp6d.p_poseNMS import pose_nms
from visual import visualize_bounding_box
from lib.utils.base_utils import Projector


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


def val(net, class_name, index):
    net.eval()
    scale=128
    linemod_path = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/'

    sift_path = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/fps/' + class_name + '.txt'
    camera_K = np.array([[572.4114, 0., 325.2611],
                         [0., 573.57043, 242.04899],
                         [0., 0., 1.]])
    train_data = LinemodDataSet(
        root='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/', \
        use_render=False, name=class_name, use_fuse=False, scale=scale)
    test_data = LinemodDataSet(
        root='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/',
        name=class_name, train=False, scale=scale)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    if class_name in ['ape', 'cat', 'can','duck', 'driller', 'eggbox','holepuncher', 'glue']:
        occ_test_data = LinemodOcclusionDataSet(
            "/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/OCCLUSION_LINEMOD/",
            name=class_name)
        occ_test_data_loader = torch.utils.data.DataLoader(occ_test_data, batch_size=1, shuffle=False, num_workers=4)

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

    eval = Evaluator(class_name)
    # print(eval.linemod_db.get_diameter('cat'))
    print("eval ", index, " epoch")
    trans = []
    pixerror = [[] for _ in range(32)]
    needidxs= [283]
    nidx=0
    for i, data in enumerate(occ_test_data_loader, 0):
        # get the inputs
        # c_data=data[0]
        # data=data[1]
        if nidx not in needidxs:
            nidx=nidx+1
            continue
        else:
            nidx=nidx+1
        c_data, data = data
        image, heatmaps, weights = [d.cuda() for d in c_data]
        ori_img,bbox, rate, farthest, farthest_3d, K, RT = data

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

        for i in range(32):
            co.append(np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))
            preds_max.append(
                np.array([b[0][i][d[0][i]].cpu().numpy().item(), d[0][i].cpu().numpy().item()], dtype=np.float32))

            maxvals.append(a[0][i][d[0][i]].cpu().numpy().item())
            # co.append([b[0][i][d[0][i]-1].cpu().numpy().item(),d[0][i]])

        # print(maxvals)
        heatmaps_pred = heatmaps_pred.cpu().detach().numpy()

        preds = get_final(heatmaps_pred, co)


        large_k=np.sum(np.asarray(maxvals)>0.8)
        #print(large_k)
        if class_name in ['ape','cat','duck','eggbox','holepuncher']:

            large_k=max(large_k,24)
        else:
            large_k=max(large_k,12)
        idxs = heapq.nlargest(large_k, range(len(maxvals)), maxvals.__getitem__)
        # print(idxs)

        '''
        #heatmaps_pred=heatmaps_pred.cpu().detach().numpy()
        heatmaps=heatmaps.cpu().detach().numpy()
        farthest=np.squeeze(farthest.cpu().detach().numpy())
        for i in range(32):

            if not i in idxs:
                continue

            print(maxvals[i])

            img=image[0]
            img=img.permute(1,2,0).detach().cpu().numpy()



            hp=heatmaps_pred[0,i]
            label=heatmaps[0,i]
            weight=weights[0,i].detach().cpu().numpy()
            

            hp=np.abs(hp)
            hp=hp/np.max(hp)
            t=np.zeros(hp.shape,dtype=np.float32)
            hp=np.abs(hp-label)
            print(np.linalg.norm(hp,ord=1))
            t=cv2.merge([hp,t,label])
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
        p2d = points_2d[idxs]

        pose_pred = pnp(p3d, p2d, camera_K, cv2.SOLVEPNP_EPNP)


        # print(pose_pred)
        stop = time.clock()
        # print(stop-begin)
        run_time.update(stop - begin)
        pose_targets=np.squeeze(RT.detach().cpu().numpy())
        bb8_3d = np.loadtxt(linemod_path+'/'+class_name+ '/corners.txt')
        run_time.update(stop - begin)
        projector = Projector()
        bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'linemod')
        bb8_2d_gt = projector.project(bb8_3d, pose_targets, 'linemod')
        rgb = ori_img.permute(0, 1,2, 3).detach().cpu().numpy()
        visualize_bounding_box(rgb, bb8_2d_pred[None, None, ...], bb8_2d_gt[None, None, ...],save=False)

        eval.evaluate(pose_pred, np.squeeze(RT.cpu().numpy()), class_name)
        
        translation_distance = np.abs(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        trans.append(translation_distance)
        print(np.mean(trans, axis=0))

        farthest = np.squeeze(farthest.detach().cpu().numpy())
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

    mean_tran = np.mean(trans, axis=0)

    #print(np.mean(trans, axis=0))
    with open('add_dis/'+class_name+'.txt','w') as fc:
        for dis in eval.add_dists:
            fc.write(str(dis)+'\n')

    print(run_time.sum, run_time.count, run_time.avg)

    print('Gaiss', dis_gauss.avg)
    print(np.mean(eval.cm))
    print(np.mean(eval.degree))

    print(index, eval.average_precision(verbose=False))
    result = eval.average_precision(verbose=False)
    ret=[class_name,str(index),str(round(result[0],5)),str(round(result[1],5)),str(round(result[2],5)),
         str(round(dis_gauss.avg,5)),str(round(np.mean(eval.cm),5)),str(round(np.mean(eval.degree),5))]
    fi = open('load/load_' + class_name + '.txt', 'a')
    fi.write(ret[0] + '\t' + ret[1] + '\t' + ret[2] + '\t'
             + ret[3] + '\t' + ret[4] + '\t' + ret[5] +'\t'
             + ret[6] + '\t' + ret[7] + '\n')
    fi.close()
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
    class_names=['cat']
    #class_names=['ape', 'benchvise','cam','can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'iron', 'lamp', 'phone']
    #class_names=[ 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp','phone']
    net = seg_hrnet2.get_seg_model(config)
    net = NetWrapper(net)
    net = DataParallel(net).cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    for class_name in class_names:
        if class_name == 'cat' or class_name == 'can' or class_name == 'driller':
            idx = 24
        else:
            idx = 29
        idx='best_add'
        load_model(net.module.net, optimizer, './net_' + class_name, idx)
        # for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        ret=val(net, class_name, index='best_add')
        with open('occ_result.txt','a') as fi:
            fi.write(ret[0] + '\t' + ret[1] + '\t' + ret[2] + '\t'
                     + ret[3] + '\t' + ret[4] + '\t' + ret[5] +'\t'
                     + ret[6] + '\t' + ret[7] + '\n')


    # load_model(net.module.net, optimizer, './net_'+class_name, 9)
    # val(net,class_name,9)
    print("123")
