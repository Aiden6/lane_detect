#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:28:38 2018

@author: aiden_leo
"""

from CNNBaseModel import CNNBaseModel
import tensorflow as tf
from collections import OrderedDict

# 基于DenseNet的编码器

class DenseEncoder(CNNBaseModel):
    """
    基于DenseNet的编码器
    """
    def __init__(self, l, n, growthrate, phase, with_bc=False,
                 bc_theta=0.5):
        """

        :param l: According to the DenseNet paper l refers to the depth of the network
        :param growthrate: According to the DenseNet paper
                           GrowthRate refers to the dense block out dims
        :param n: According to the DenseNet n refers to the block nums of the network
        :param with_bc: whether to use BC in the model
        :param phase: is training or testing
        :param bc_theta: transmition theta thresholding
        """
        super(DenseEncoder, self).__init__()
        self._L = l
        self._block_depth = int((l - n - 1) / n)
        self._N = n
        self._growthrate = growthrate
        self._with_bc = with_bc
        self._phase = phase
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()
        self._bc_theta = bc_theta
        return
    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)
    def __str__(self):
        encoder_info = 'A densenet with net depth: {:d} block nums: ' \
                       '{:d} growth rate: {:d} block depth: {:d}'.\
            format(self._L, self._N, self._growthrate, self._block_depth)
        return encoder_info   
    #block 块中的混合卷积
    #包括BN,relu,conv操作
    def _composite_conv(self, inputdata, out_channel, name):

        with tf.variable_scope(name):

            bn_1 = self.layerbn(inputdata=inputdata, is_training=self._is_training, name='bn_1')

            relu_1 = self.relu(bn_1, name='relu_1')

            if self._with_bc:
                conv_1 = self.conv2d(inputdata=relu_1, out_channel=out_channel,
                                     kernel_size=1,
                                     padding='SAME', stride=1, use_bias=False,
                                     name='conv_1')

                bn_2 = self.layerbn(inputdata=conv_1, is_training=self._is_training, name='bn_2')

                relu_2 = self.relu(inputdata=bn_2, name='relu_2')
                conv_2 = self.conv2d(inputdata=relu_2, out_channel=out_channel,
                                     kernel_size=3,
                                     stride=1, padding='SAME', use_bias=False,
                                     name='conv_2')
                return conv_2
            else:
                conv_2 = self.conv2d(inputdata=relu_1, out_channel=out_channel,
                                     kernel_size=3,
                                     stride=1, padding='SAME', use_bias=False,
                                     name='conv_2')
                return conv_2
    #block块中的通道合并操作
    def _denseconnect_layers(self, inputdata, name):
        with tf.variable_scope(name):
            conv_out = self._composite_conv(inputdata=inputdata,
                                            name='composite_conv',
                                            out_channel=self._growthrate)
            concate_cout = tf.concat(values=[conv_out, inputdata], axis=3,
                                     name='concatenate')
        return concate_cout
    #block块与块之间的转换层，包括bn,１＊１conv和２＊２average_pool
    def _transition_layers(self, inputdata, name):
        input_channels = inputdata.get_shape().as_list()[3]
        with tf.variable_scope(name):
            # First batch norm
            bn = self.layerbn(inputdata=inputdata, is_training=self._is_training, name='bn')

            # Second 1*1 conv
            if self._with_bc:
                out_channels = int(input_channels * self._bc_theta)
                conv = self.conv2d(inputdata=bn, out_channel=out_channels,
                                   kernel_size=1, stride=1, use_bias=False,
                                   name='conv')
                # Third average pooling
                avgpool_out = self.avgpooling(inputdata=conv, kernel_size=2,
                                              stride=2, name='avgpool')
                return avgpool_out
            else:
                conv = self.conv2d(inputdata=bn, out_channel=input_channels,
                                   kernel_size=1, stride=1, use_bias=False,
                                   name='conv')
                # Third average pooling
                avgpool_out = self.avgpooling(inputdata=conv, kernel_size=2,
                                              stride=2, name='avgpool')
                return avgpool_out
    #一个block块的组成
    def _dense_block(self, inputdata, name):
        block_input = inputdata
        with tf.variable_scope(name):
            for i in range(self._block_depth):
                block_layer_name = '{:s}_layer_{:d}'.format(name, i + 1)
                block_input = self._denseconnect_layers(inputdata=block_input,
                                                        name=block_layer_name)
        return block_input
    #densenet网络
    def encode(self, input_tensor, name):

        encode_ret = OrderedDict()

        # First apply a 3*3 16 out channels conv layer
        # mentioned in DenseNet paper Implementation Details part
        with tf.variable_scope(name):
            #先卷积
            conv1 = self.conv2d(inputdata=input_tensor, out_channel=16,
                                kernel_size=3, use_bias=False, name='conv1')
            dense_block_input = conv1

            # Second apply dense block stage
            for dense_block_nums in range(self._N):
                dense_block_name = 'Dense_Block_{:d}'.format(dense_block_nums + 1)

                # dense connectivity
                dense_block_out = self._dense_block(inputdata=dense_block_input,
                                                    name=dense_block_name)
                # apply the trainsition part
                dense_block_out = self._transition_layers(inputdata=dense_block_out,
                                                          name=dense_block_name)
                dense_block_input = dense_block_out
                encode_ret[dense_block_name] = dict()
                encode_ret[dense_block_name]['data'] = dense_block_out
                encode_ret[dense_block_name]['shape'] = dense_block_out.get_shape().as_list()

        return encode_ret
if __name__ == '__main__':
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 384, 1248, 3], name='input_tensor')
    encoder = DenseEncoder(l=100, growthrate=16, with_bc=True, phase=tf.constant('train'), n=5)
    ret = encoder.encode(input_tensor=input_tensor, name='Dense_Encode')
    for layer_name, layer_info in ret.items():
        print('layer_name: {:s} shape: {}'.format(layer_name, layer_info['shape']))