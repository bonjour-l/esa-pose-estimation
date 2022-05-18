# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from collections import namedtuple
import heapq
from data_load4 import LinemodDataSet
from demo2 import val
from net import Resnet18_8s, Resnet50_8s
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from models import seg_hrnet2
from config import config
import time
import os
import torch
from torch import nn
import torch.optim as optim
from loss import focal_l2_loss

from loss import Smooth_l1
from loss import Loss_weighted,WLoss
from logger import Logger
from evaluation import AverageMeter
import numpy as np
from inference import get_final, get_final2
from evaluation import AverageMeter, Evaluator
from pnp import pnp


class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.mseloss = nn.MSELoss(reduce=False)
        self.sl1loss = Loss_weighted()
        self.hwing_loss = Loss_weighted()
        self.tloss = nn.MSELoss(reduce=False)
        self.weight = torch.tensor([1, 1, 5], device='cuda:0')
        self.wloss=WLoss()

    def forward(self, image, heatmaps, weights, tran, train=True):
        start = time.clock()
        heatmaps_pred, xt = self.net(image)
        # print(time.clock()-start)

        if not train:
            return heatmaps_pred, 0

        loss_vertex = self.hwing_loss(heatmaps_pred, heatmaps, weights)

        # loss_vertex = self.awing_loss(heatmaps_pred, heatmaps, weights)
        loss_vertex = torch.mean(loss_vertex.view(loss_vertex.shape[0], -1), 1)
        xt0=xt[:,:2]
        xt1=xt[:,2]
        tran0=tran[:,:2]
        tran1=tran[:,2]
        loss_tran0=self.tloss(xt0,tran0)
        loss_tran1=5*self.wloss(xt1,tran1)

        loss_tran=torch.cat([loss_tran0,torch.unsqueeze(loss_tran1,dim=1)],dim=1)

        return heatmaps_pred, xt, loss_vertex, loss_tran


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
                    class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                        torch.prod(
                            torch.LongTensor(list(module.weight.data.size()))) *
                        torch.prod(
                            torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]
            try:
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size=list(input[0].size()),
                        output_size=list(output.size()),
                        num_parameters=params,
                        multiply_adds=flops)
                )
            except Exception as e:
                pass

        if not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.Sequential) \
                and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
                  os.linesep + \
                  "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                      ' ' * (space_len - len("Name")),
                      ' ' * (space_len - len("Input Size")),
                      ' ' * (space_len - len("Output Size")),
                      ' ' * (space_len - len("Parameters")),
                      ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                  + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                       + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
               + "Total Parameters: {:,}".format(params_sum) \
               + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(
        flops_sum / (1024 ** 3)) \
               + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def save_model(net, optim, epoch, model_dir, save_name):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(save_name)))


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
        pth = 'last'
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    model.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load model {} epoch {}'.format(model_dir, pretrained_model['epoch']))
    return pretrained_model['epoch'] + 1


def adjust_learning_rate(optimizer, epoch, lr_decay_begin, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        if ((epoch + 1) < lr_decay_begin):
            return lr_before

        if ((epoch + 1) % lr_decay_epoch) != 0:
            return lr_before

        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)
    print('changing learning rate {:5f} to {:.5f}'.format(lr_before, max(param_group['lr'], min_lr)))
    return max(param_group['lr'], min_lr)


def adjust_learning_rate(optimizer, epoch, epoch_list, lr_list):
    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']

        if epoch in epoch_list:
            idx = epoch_list.index(epoch)
            param_group['lr'] = lr_list[idx]
            # param_group['lr'] = param_group['lr'] * 0.1

    print('changing learning rate {:5f} to {:.5f}'.format(lr_before, param_group['lr']))
    return param_group['lr']


def train(class_name):
    camera_K = np.array([[572.4114, 0., 325.2611],
                         [0., 573.57043, 242.04899],
                         [0., 0., 1.]])
    f = open('log/log_' + class_name + '.txt', 'a')
    f.close()
    net = seg_hrnet2.get_seg_model(config)
    # net = Resnet18_8s(ver_dim=32)

    dump_input = torch.rand(
        (1, 3, 128, 128)
    )

    # details=get_model_summary(net, dump_input)
    # print(details)
    net = NetWrapper(net)
    net = DataParallel(net).cuda()
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    lr_start = 0.0001

    heatmap_rate = 0.5

    train_data = LinemodDataSet(
        root='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/',
        use_render=True,
        use_fuse=True,
        name=class_name, scale=128,
        gauss_size=2)
    test_data = LinemodDataSet(
        root='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/',
        name=class_name, scale=128, train=False)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.Adam(net.parameters(), lr=lr_start)

    if False:
        begin_epoch = 0

        logger = Logger('log/log_' + class_name + '.txt', resume=True)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'ver loss', 'heap loss'])
    else:
        begin_epoch = load_model(net.module.net, optimizer, './net2_' + class_name, epoch=-1)
        logger = Logger('log/log_' + class_name + '.txt', resume=True)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'heatmap loss', 'tran loss'])

    losses_hm = AverageMeter()
    losses_tran = AverageMeter()
    losses = AverageMeter()
    run_time = AverageMeter()

    run_time.reset()

    # epoch_list = [100,150, 170,190]
    # lr_list=[lr_start/2,lr_start/10,lr_start/20,lr_start/100]

    epoch_list = [30, 40, 170]  ###
    lr_list = [lr_start / 10, lr_start / 100, lr_start / 1000]
    best_val = [999, 999]
    hmrate=0
    for epoch in range(begin_epoch, 40):  # loop over the dataset multiple times
        # val(net,class_name,epoch)
        '''
        for i,data in enumerate(test_data_loader, 0):

            c_data,data=data
            image, mask,xt, heatmaps,weights= [d.cuda() for d in c_data]
            bbox,rate,farthest,farthest_3d,K,RT  = data



            heatmaps_pred,xt, loss_vertex,loss_tran = net(image,heatmaps,weights,xt,train=False)


            x,y,w,h=[np.squeeze(d.numpy()) for d in bbox]


            heatmaps_pred=heatmaps_pred.detach()
            a,b=torch.max(heatmaps_pred,dim=3)
            c,d=torch.max(a,dim=2)


            co=[]

            maxvals=[]

            for i in range(32):
                co.append(np.array([b[0][i][d[0][i]].cpu().numpy().item(),d[0][i].cpu().numpy().item()],dtype=np.float32))

                maxvals.append(a[0][i][d[0][i]].cpu().numpy().item())



            start=time.clock()
            heatmaps_pred=heatmaps_pred.cpu().detach().numpy()

            preds=get_final2(heatmaps_pred,co)

            stop=time.clock()

            idxs=heapq.nlargest(24,range(len(maxvals)),maxvals.__getitem__)

            ori_preds=preds*(1/rate.numpy())+(x,y)


            points_3d=np.squeeze(farthest_3d.detach().cpu().numpy())

            points_2d=ori_preds

            p3d=points_3d[idxs]
            p2d=points_2d[idxs]


            pose_pred = pnp(p3d, p2d, camera_K,cv2.SOLVEPNP_EPNP)

            #print(time.clock()-start)
            run_time.update(stop-start)
        print(run_time.sum,run_time.count,run_time.avg)
         '''
        net.train()
        lr = adjust_learning_rate(optimizer, epoch, epoch_list, lr_list)
        if epoch in epoch_list:
            save_model(net.module.net, optimizer, epoch, './net2_' + class_name, str(epoch))

        l1 = l2 = running_loss = 0.0
        losses_hm.reset()
        losses_tran.reset()

        for i, data in enumerate(train_data_loader, 0):
            c_data, data = data
            image, tran, heatmaps, weights = [d.cuda() for d in c_data]

            # get the inputs

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            heatmaps_pred, xt, loss_hm, loss_tran = net(image, heatmaps, weights, tran)

            loss_hm = torch.mean(loss_hm)
            loss_tran = torch.mean(loss_tran)
            #hmrate=max(0.1,0.1*(21-epoch))
            loss = loss_hm + loss_tran

            loss.backward()
            optimizer.step()

            losses_hm.update(loss_hm.detach(), 1)
            losses_tran.update(loss_tran.detach(), 1)
            losses.update(loss.detach(), 1)

            # print statistics
            l1 += loss_hm.item()
            l2 += loss_tran.item()
            running_loss += loss.item()

            if i % 10 == 9:  # print every 2000 mini-batches
                print(class_name, '[%d, %5d] loss : %.6f loss_heatmap : %.6f loss_tran : %.6f ' % \
                      (epoch + 1, i + 1, running_loss / 10, l1 / 10, l2 / 10))

                l1 = l2 = running_loss = 0.0
        # for i, data in enumerate(train_data_loader, 0):
        #     c_data,data=data
        #     image, mask, heatmaps,weights= [d.cuda() for d in c_data]
        #     # get the inputs
        #
        #     # zero the parameter gradients
        #     optimizer.zero_grad()
        #
        #
        #     # forward + backward + optimize
        #     seg_pred, heatmaps_pred, loss_seg, loss_vertex = net(image,mask.long(),heatmaps,weights)
        #     loss_seg=torch.mean(loss_seg)
        #     loss_vertex=torch.mean(loss_vertex)
        #
        #     loss = (1-heatmap_rate)*loss_seg+heatmap_rate*loss_vertex
        #
        #     eval_losses_seg.update(loss_seg,image.size(0))
        #     eval_losses_ver.update(loss_vertex,image.size(0))
        #     eval_losses.update(loss,image.size(0))

        logger.append([epoch + 1, lr, losses.avg, losses_hm.avg, losses_tran.avg])

        save_model(net.module.net, optimizer, epoch, './net2_' + class_name, 'last')
        if epoch >5:
            #save_model(net.module.net, optimizer, epoch, './net2_' + class_name, str(epoch))
            result = val(net, class_name, epoch)
            if result[0] < best_val[0]:
                save_model(net.module.net, optimizer, epoch, './net2_' + class_name, 'best_tran')
                best_val[0] = result[0]
            if result[1] < best_val[1]:
                save_model(net.module.net, optimizer, epoch, './net2_' + class_name, 'best_rotate')
                best_val[1] = result[1]

    logger.close()
    print('Finished Training')


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print('Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # class_names=['iron','driller','lamp','eggbox','holepuncher']
    # class_names=['cat','ape','cam','duck','can','iron','phone','benchvise','driller','lamp','holepuncher','eggbox','glue']
    class_names = ['cat','ape','benchvise','cam','can']
    for class_name in class_names:
        train(class_name)
        try:

            train(class_name)
        except Exception as e:
            print(e)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
