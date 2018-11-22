#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:56:04 2018

@author: aiden_leo
"""

import datetime

oldtime=datetime.datetime.now() 

import os
file_array = []
print(os.getcwd()) #获取当前工作目录路径
print("'q' for quit & without delete images, 'c' for capture image, 'd' for quit & delete all images")
#
image_num = 0
# 打开摄像头并灰度化显示

import cv2
capture = cv2.VideoCapture(0)
# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
width, height = capture.get(3), capture.get(4)
print(width, height)
# 以原分辨率的一倍来捕获
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))


while(capture.isOpened()):

    key = cv2.waitKey(1)
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
    if ret:
        outfile.write(frame)  # 写入文件  录制视频
        cv2.imshow('frame', frame)
    else:
        break
    
    if (datetime.datetime.now()- oldtime).microseconds > 200:
        cv2.imwrite(str(os.getcwd())+'/'+str(image_num)+'.jpg',frame)
        image_num += 1
        oldtime=datetime.datetime.now() 
    
    if key == ord('q'):
        print("在工作文件夹下有%d张图片" % image_num)
        capture.release()
        outfile.release()
        cv2.destroyAllWindows()
        break
    elif key  == ord('c'):
        cv2.imwrite(str(os.getcwd())+'/'+str(image_num)+'.jpg',frame)
        image_num+=1
    elif key == ord('d'):
        capture.release()
        outfile.release()
        cv2.destroyAllWindows()
        
        f_list = os.listdir(os.getcwd())
        for fileNAME in f_list:
            # os.path.splitext():分离文件名与扩展名
            if os.path.splitext(fileNAME)[1] == '.jpg':
                file_array.append(fileNAME )
        # 以上是从pythonscripts文件夹下读取所有excel表格，并将所有的名字存储到列表filearray
        print("在工作文件夹下有%d张图片，同时已经被删除" % len(file_array))
        for i in range(len(file_array)):
            f_name = file_array[i]
            os.remove(f_name)
        break
capture.release()
outfile.release()
cv2.destroyAllWindows()



