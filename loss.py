import math

import torch
from scipy import ndimage
from torch import nn
import numpy as np


def focal_l2_loss(s, sxing, mask_miss, gamma=2, nstack_weight=[1, 1, 1, 1]):
    """
    Compute the focal L2 loss between predicted and groundtruth score maps.
    :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
    :param sxing: target tensor (nstack, batch, channel, height, width)
    :param mask_miss: tensor (1, batch, 1, height, width)
    :param gamma: focusing parameter
    :return: a scalar tensor
    """
    # eps = 1e-8  # 1e-12
    # s = torch.clamp(s, eps, 1. - eps)  # improve the stability of the focal loss

    st = torch.where(torch.ge(sxing, 0.01), s, 1 - s)

    factor = (1. - st) ** gamma

    # multiplied by mask_miss via broadcast operation
    out = (s - sxing) ** 2 * factor * mask_miss  # type: torch.Tensor

    # sum over the feature map, should divide by batch_size afterwards
    loss = out.sum(dim=(1, 2, 3))

    return loss
    loss_nstack = out.sum(dim=(1, 2, 3, 4))  # losses from nstack 1, 2, 3, 4...
    assert len(loss_nstack) == len(nstack_weight), nstack_weight
    print(' heatmap focal L2 loss per stack..........  ', loss_nstack.detach().cpu().numpy())
    weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
    loss = sum(weight_loss) / sum(nstack_weight)
    return loss


class AWing(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y))) * (self.alpha - y) * (
                    (self.theta / self.epsilon) ** (self.alpha - y - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon) ** (self.alpha - y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]

        return lossMat

class HeatmapWing(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, epsilon=2.0, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y))) * (self.alpha - y) * (
                (self.theta / self.epsilon) ** (self.alpha - y - 1)) / self.epsilon
        C = self.theta  - self.omega * torch.log(1 + (self.theta / (self.epsilon-y)) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / (self.epsilon- y[case1_ind])) ** (self.alpha - y[case1_ind]))
        lossMat[case2_ind] = torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]

        return lossMat



class Smooth_l1(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 0.5

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = 0.5 * (y[case1_ind] - y_pred[case1_ind]) ** 2
        lossMat[case2_ind] = abs(y[case2_ind] - y_pred[case2_ind]) - 0.375
        return lossMat



class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2,theta=0.5):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.theta=theta

    def forward(self, y_pred, y):

        lossMat = torch.zeros_like(y_pred)
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        C = self.theta - self.omega * torch.log(torch.tensor(1) + self.theta / self.epsilon)
        lossMat[case1_ind] = self.omega * torch.log(1 + torch.abs(y[case1_ind] - y_pred[case1_ind]) / self.epsilon)
        lossMat[case2_ind] = abs(y[case2_ind] - y_pred[case2_ind]) - C
        return lossMat

class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)
        self.sl1 = Smooth_l1()
        self.Hwing=HeatmapWing(alpha, omega, 2, theta)
        self.wing=WingLoss()
        self.mse=nn.MSELoss()
    def forward(self, y_pred, y, M):
        M = M.float()
        Loss = self.Hwing(y_pred,y)
        weighted = Loss * (self.W * M + 1.)
        return weighted



def generate_weight_map(heatmap):
    k_size = 3
    dilate = ndimage.grey_dilation(heatmap, size=(k_size, k_size))
    weight_map = heatmap
    weight_map[np.where(dilate > 0.2)] = 1
    return weight_map
    #return dilate





class WLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):

        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        lossMat = self.omega * torch.log(1 + torch.abs(pred - target) / self.epsilon)

        return lossMat
