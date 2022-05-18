from scipy.io import loadmat
import torch
from torch import nn


import numpy as np

def twoD_Gaussian(meshgrid, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    meshgrid: 包含x和y，能够表示网格所有的索引(i,j)
    xo, yo: 高斯分布的中心位置
    amplitude: 高斯分布的最大强度
    sigma_x, sigma_y: 影响高斯光斑的长宽
    theta: 控制角度，如果是一个椭圆，数值范围0-PI
    offset: 补偿值，可以使所有的数值都是正数
    '''
    x, y = meshgrid
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                      + c*((y-yo)**2)))
    # 注意返回值最好要ravel，不然curve_fit会报错
    return g.ravel()

from scipy.optimize import curve_fit

# Create x and y indices
x = np.linspace(0, 12, 13)
y = np.linspace(0, 12,13)
x, y = np.meshgrid(x, y)  # meshgrid生成了一整张索引map
#create data
data = twoD_Gaussian((x, y), 10, 3, 3, 2, 2, 0, 10)
# add some noise to the data and try to fit the data generated beforehand
initial_guess = (10, 3, 3, 2, 2, 0, 10)
data_noisy = data + 0.2*np.random.normal(size=data.shape)
# 方便起见，构造一个带噪声的数据

import time
begin=time.clock()
popt, pcov = curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)
stop=time.clock()
print((stop-begin)*32)
print(popt)
# 需要拟合的参数直接全部就放到p0去了