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
#import cpnp


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
        self.sl1loss=Loss_weighted()
        self.hwing_loss = Loss_weighted()
        self.tloss=nn.SmoothL1Loss(reduce=False)
        self.weight=torch.tensor([1,1,2],device='cuda:0')


    def forward(self, image,  heatmaps, weights,tran,train=True):

        start=time.clock()
        heatmaps_pred ,xt= self.net(image)
        #print(time.clock()-start)

        if not train:
            return heatmaps_pred,0

        loss_vertex=self.hwing_loss(heatmaps_pred, heatmaps,weights)

        #loss_vertex = self.awing_loss(heatmaps_pred, heatmaps, weights)
        loss_vertex = torch.mean(loss_vertex.view(loss_vertex.shape[0], -1), 1)
        loss_tran = self.tloss(xt,tran)
        loss_tran=torch.mean(loss_tran,dim=0)
        loss_tran=torch.mul(loss_tran,self.weight)
        return heatmaps_pred,xt,loss_vertex,loss_tran


def val(net, class_name, index):
    net.eval()
    scale=128
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
    if class_name in ['ape', 'cat', 'duck', 'driller', 'holepuncher', 'glue']:
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

    # load_model(net.module.net, optimizer, './net2_'+class_name, index*10-1)
    # load_model(net.module.net, optimizer, './net2_cat_hwing_0.2', index*10-1)

    eval = Evaluator(class_name)
    # print(eval.linemod_db.get_diameter('cat'))
    print("eval ", index, " epoch")
    trans = []
    pixerror = [[] for _ in range(32)]
    for i, data in enumerate(test_data_loader, 0):
        # get the inputs
        # c_data=data[0]
        # data=data[1]

        c_data, data = data
        image, tran,heatmaps, weights = [d.cuda() for d in c_data]
        bbox, rate, farthest, farthest_3d, K, RT = data

        begin = time.clock()

        heatmaps_pred,xt, loss_vertex,loss_tran = net(image,  heatmaps, weights,tran)
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



        idxs = heapq.nlargest(24, range(len(maxvals)), maxvals.__getitem__)
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
        '''
        tran = pose_pred[:, 3]
        r = pose_pred[:, 0:3]
        r_m, _ = cv2.Rodrigues(r)
        camera = np.concatenate([np.squeeze(r_m), tran], axis=0)
        pose = cpnp.cpnp(p3d, p2d, camera_K, camera.copy())
        # print(camera)
        # print(pose)
        tc = pose[3:]
        rc = pose[:3]
        rc_m, _ = cv2.Rodrigues(rc)
        '''
        r = pose_pred[:, 0:3]

        t=np.squeeze(xt.cpu().detach().numpy())
        t0=np.squeeze(tran.cpu().detach().numpy())

        tz=t[2]*np.squeeze(rate).numpy()
        center=t[:2]

        K=np.squeeze(K.cpu().detach().numpy())
        center[0]=(scale/2.0)-center[0]*scale
        center[1]=(scale/2.0)-center[1]*scale
        center=center * (1 / rate.numpy())+[x,y]
        center[0]=(center[0]-K[0][2])*tz/(K[0][0])
        center[1]=(center[1]-K[1][2])*tz/(K[1][1])
        t=np.asarray([center[0],center[1],tz])


        #trans.append(np.abs(t-t0)*100)




        pose_pred = np.concatenate((r, np.asarray(t.reshape(3, 1))), axis=1)
        # print(pose_pred)
        stop = time.clock()
        # print(stop-begin)
        run_time.update(stop - begin)
        eval.evaluate(pose_pred, np.squeeze(RT.cpu().numpy()), class_name)
        pose_targets = np.squeeze(RT.cpu().numpy())
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
        # fc.close()

        # eval.evaluate(ori_preds,np.squeeze(RT.cpu().numpy()),class_name,idxs,vote_type=10)

        # print(eval.average_precision(verbose=False))
    mean_tran = np.mean(trans, axis=0)
    # fc=open('record.txt','a')
    # fc.write(class_name+'\t'+str(mean_tran[0])+'\t'+str(mean_tran[1])+'\t'
    # +str(mean_tran[2])+'\t'+str(np.mean(eval.degree))+'\n')
    # print(np.mean(eval.cm))
    # print(np.mean(eval.degree))
    # fc.close()

    #print(np.mean(trans, axis=0))


    print(run_time.sum, run_time.count, run_time.avg)

    print('Gaiss', dis_gauss.avg)
    print(np.mean(eval.cm))
    print(np.mean(eval.degree))

    print(index, eval.average_precision(verbose=False))
    result = eval.average_precision(verbose=False)
    fi = open('load/load_' + class_name + '.txt', 'a')
    fi.write(class_name + '\t' + str(index) + '\t' + str(result[0]) + '\t' + str(result[1]) + '\t' + str(result[2]) + '\t'
             + str(dis_gauss.avg) + '\t' + str(np.mean(eval.cm)) + '\t' + str(np.mean(eval.degree))+'\n')
    fi.close()
    # plt.imshow(image[0].cpu().permute(1,2,0))
    # plt.show()
    return [np.mean(eval.cm),np.mean(eval.degree)]


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
    #class_names = ['ape', 'benchvise', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    # class_names=['ape','cat','duck','driller','holepuncher','glue']
    # net=Resnet50_8s(ver_dim=32)
    class_names=['ape','benchvise','cam']
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
        idx=-1
        load_model(net.module.net, optimizer, './net2_' + class_name, idx)
        # for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        val(net, class_name, index=-1)

    # load_model(net.module.net, optimizer, './net2_'+class_name, 9)
    # val(net,class_name,9)
    print("123")
