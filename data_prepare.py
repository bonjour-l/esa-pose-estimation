import json
import pycocotools
import cv2
import numpy as np
import numpy.ma as ma
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

real_data=dict()
render_data=dict()
fuse_data=dict()

with open("data/cat_real.pkl", 'rb') as fo:     # 读取pkl文件数据
    real_data = pickle.load(fo, encoding='bytes')

with open("data/cat_render.pkl", 'rb') as fo:     # 读取pkl文件数据
    render_data = pickle.load(fo, encoding='bytes')

with open("data/cat_fuse.pkl", 'rb') as fo:     # 读取pkl文件数据
    fuse_data = pickle.load(fo, encoding='bytes')



print("123")