import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import cv2
# 数据集目录 #
filepath = '/home/aiden_leo/Capture_video/Capture_image_video_by_opencv'
file_array = []
f_list = os.listdir(filepath)
#f_list = os.listdir(os.getcwd())
R_channel = 0
G_channel = 0
B_channel = 0

for fileNAME in f_list:
    # os.path.splitext():分离文件名与扩展名
    if os.path.splitext(fileNAME)[1] == '.jpg':
        file_array.append(fileNAME)
# 以上是从pythonscripts文件夹下读取所有excel表格，并将所有的名字存储到列表filearray
print("在工作文件夹下有%d张图片" % len(file_array))

for i in range(len(file_array)):
    f_name = file_array[i]
    #img = imread(os.path.join(filepath, f_name))
    img = cv2.imread(os.path.join(filepath, f_name), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(file_array)*512*256  # 这里（384,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num
print("R_mean is %f, G_mean is %f, B_mean is %f" %(R_mean, G_mean, B_mean))
