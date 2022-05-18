# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import cv2
from transforms import transform_preds
import torch




def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    '''
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    '''
    return preds, maxvals


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def my_taylor(hm, coord):
    sigma=2
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        #hm=abs(hm)

        hx  = 0.5 * (math.log(hm[py][px+1]) - math.log(hm[py][px-1]))
        hy  = 0.5 * (math.log(hm[py+1][px]) - math.log(hm[py-1][px]))
        hxx = 0.25 * (math.log(hm[py][px+2]) - 2 * math.log(hm[py][px]) + math.log(hm[py][px-2]))
        hyy = 0.25 * (math.log(hm[py+2][px]) - 2 * math.log(hm[py][px]) + math.log(hm[py-2][px]))
        if hxx!=0 and hyy!=0:
            offset=[-hx/hxx,-hy/hyy]
            #print(hppx,hppy)
            #print(offset)
            if offset[0]<1 and offset[1]<1:
                coord += offset
    return coord

def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i,j] = dr[border: -border, border: -border].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
    return hm


def get_final_preds(hm, center, scale):
    coords, maxvals = get_max_preds(hm)


    # post-processing
    #hm = gaussian_blur(hm, 11)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n,p] = my_taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    return preds, maxvals
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )
    
    return preds, maxvals

def get_final(hm,coords):


    # post-processing
    #hm = gaussian_blur(hm, 11)
    hm = np.maximum(hm, 1e-10)
    #hm = np.log(hm)

    #hm=torch.maximum(hm,torch.tensor(1e-10))
    #hm=torch.log(hm)

    for p in range(len(coords)):
    	coords[p] = my_taylor(hm[0][p], coords[p])

    preds = coords.copy()

    return preds

def get_final2(hm,coords):


    # post-processing
    hm = gaussian_blur(hm, 11)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)

    #hm=torch.maximum(hm,torch.tensor(1e-10))
    #hm=torch.log(hm)

    for p in range(len(coords)):
        coords[p] = taylor(hm[0][p], coords[p])

    preds = coords.copy()

    return preds
def getPrediction(hms,  inpH=128, inpW=128):
    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    return preds,  maxval
    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(
                round(float(preds[i][j][1])))
            # if 1 < pX < opt.outputResW - 2 and 1 < pY < opt.outputResH - 2:
            if 0 < pX < inpH - 1 and 0 < pY < inpH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                diff = diff.sign() * 0.25
                diff[1] = diff[1] * inpH / inpW
                preds[i][j] += diff.cuda()



    return preds,  maxval


