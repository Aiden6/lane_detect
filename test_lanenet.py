#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
# 命令行解析包
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log  # 日志输出 google的开源日志系统
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

import lane_detect_merge_model
import lane_detect_cluster
import lane_detect_postprocess
import global_config



CFG = global_config.cfg

#VGG_MEAN = [103.939, 116.779, 123.68] # 原来的
VGG_MEAN = [117.582, 119.286, 115.697]
#定义一个初始化参数的函数
def init_args():


    # 帮助信息　-h
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()
# 单张进行测试
# 定义测试函数
# 输入图片路径　权重路径　是否使用GPU
def test_lanenet(image_path, weights_path, use_gpu):

    #断言　不存在即返回该路径not exist
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    # 输出读取信息
    log.info('开始读取图像数据并进行预处理')
    # 计时
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    # 图片裁剪
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    # 图像减去一个列表，每个通道减去一个均值　去均值
    image = image - VGG_MEAN
    # 读取图片用时
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    # 输入张量定义
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    # 训练阶段定义
    phase_tensor = tf.constant('train', tf.string)
    # 模块类LaneNet的对象初始化
    net = lane_detect_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    # 调用对象方法inference 返回二进制分割结果和实例分割结果
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')
    # 定义进行聚类的对象cluster　对结果进行聚类
    cluster = lane_detect_cluster.LaneNetCluster()
    # 定义后处理类的对象　对结果进行后处理
    postprocessor = lane_detect_postprocess.LaneNetPoseProcessor()
    # 定义保存和恢复权重的对象
    saver = tf.train.Saver()
    # 是否使用GPU
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # 占用显存的比例
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    # 显存慢慢增加　按需设置显存
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    # 加速算法
    sess_config.gpu_options.allocator_type = 'BFC'
    # 会话设置
    sess = tf.Session(config=sess_config)
    # with上下文打开会话，会自动关闭。
    with sess.as_default():
        # 加载参数
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        # fetch binary_seg_ret&instance_seg_ret;输入图像张量
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        # 提示信息
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        # 聚类结果
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])
        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=instance_seg_image[0])
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)

        ele_mex = np.max(instance_seg_image[0], axis=(0, 1))
        for i in range(3):
            if ele_mex[i] == 0:
                scale = 1
            else:
                scale = 255 / ele_mex[i]
            instance_seg_image[0][:, :, i] *= int(scale)
        embedding_image = np.array(instance_seg_image[0], np.uint8)
        # cv2.imwrite('embedding_mask.png', embedding_image)

        # mask_image = cluster.get_lane_mask_v2(instance_seg_ret=embedding_image)
        # mask_image = cv2.resize(mask_image, (image_vis.shape[1], image_vis.shape[0]),
        #                         interpolation=cv2.INTER_LINEAR)

        cv2.imwrite('binary_ret.png', binary_seg_image[0] * 255)
        cv2.imwrite('instance_ret.png', embedding_image)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()

    return

# batch测试网络
def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    # 图片地址是否存在
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)
    # 输出读取信息
    log.info('开始获取图像文件路径...')
    # 返回匹配的路径列表
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)
    # 输入张量占位
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('train', tf.string)
    # 模块类LaneNet的对象初始化
    net = lane_detect_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    # 推理模型 # 调用对象方法inference 返回二进制分割结果和实例分割结果
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_loss')
    # 定义进行聚类的对象cluster　对结果进行聚类
    cluster = lane_detect_cluster.LaneNetCluster()
    # 后处理
    postprocessor = lane_detect_postprocess.LaneNetPoseProcessor()
    # 保存模型参数
    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    # 启动会话
    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        # 加入batch后的计算代数
        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):
            # 显示信息
            log.info('[Epoch:{:d}] 开始图像读取和预处理...'.format(epoch))
            # 计时
            t_start = time.time()
            # 批量读入并裁剪
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            # 计时
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预处理{:d}张图像, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            # 运行会话
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预测{:d}张图像车道线, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)
                    # log.info('[Epoch:{:d}] Detection image {:s} complete'.format(epoch, image_name))
            log.info('[Epoch:{:d}] 进行{:d}张图像车道线聚类, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    # 判断存储路径是否存在，若不存在显示错误并创建
    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
