import numpy as np
import torch
import scipy.misc
from torchsample.transforms import SpecialCrop, Pad
import torch.nn.functional as F
import cv2


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def drawGaussian(img, pt, sigma):
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    center = torch.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = torch.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt.int()


def transformBoxInvert(pt, ul, br, inpH, inpW, resH, resW):
    center = torch.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = (pt * lenH) / resH
    _pt[0] = _pt[0] - max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] - max(0, (lenH - 1) / 2 - center[1])

    new_point = torch.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point


def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = br.int()
    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    newDim = torch.IntTensor((img.size(0), int(lenH), int(lenW)))
    newImg = img[:, ul[1]:, ul[0]:].clone()
    # Crop and Padding
    size = torch.IntTensor((int(br[1] - ul[1]), int(br[0] - ul[0])))
    newImg = SpecialCrop(size, 1)(newImg)
    newImg = Pad(newDim)(newImg)
    # Resize to output
    v_Img = torch.autograd.Variable(newImg)
    v_Img = torch.unsqueeze(v_Img, 0)
    # newImg = F.upsample_bilinear(v_Img, size=(int(resH), int(resW))).data[0]
    if torch.__version__ == '0.4.0a0+32f3bf7' or torch.__version__ == '0.4.0':
        newImg = F.upsample(v_Img, size=(int(resH), int(resW)),
                            mode='bilinear', align_corners=True).data[0]
    else:
        newImg = F.interpolate(v_Img, size=(int(resH), int(resW)),
                               mode='bilinear', align_corners=True).data[0]
    return newImg


def flip_v(x, cuda=False):
    x = flip(x.cpu().data)
    if cuda:
        x = x.cuda()
    x = torch.autograd.Variable(x)
    return x


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    # dim = x.dim() - 1
    x = x.numpy().copy()
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(
                np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    # x = x.swapaxes(dim, 0)
    # x = x[::-1, ...]
    # x = x.swapaxes(0, dim)

    return torch.from_numpy(x.copy())


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.dim() == 3 or x.dim() == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
            #x[:, dim0], x[:, dim1] = deepcopy((x[:, dim1], x[:, dim0]))
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
            #x[dim0], x[dim1] = deepcopy((x[dim1], x[dim0]))
    return x


def shuffleLR_v(x, dataset, cuda=False):
    x = shuffleLR(x.cpu().data, dataset)
    if cuda:
        x = x.cuda()
    x = torch.autograd.Variable(x)
    return x
