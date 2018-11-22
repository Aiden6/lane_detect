# -*- coding: utf-8 -*-

"""
实现LaneNet中的二分类图像分割模型
"""

import tensorflow as tf

# 从模块中导入类
from CNNBaseModel import CNNBaseModel
from vgg16_net_encoder import VGG16Encoder
from dense_net_encoder import DenseEncoder
from fcn_net_decoder import FCNDecoder


class LaneNetBinarySeg(CNNBaseModel):

    """
    实现语义分割模型
    """
    def __init__(self, phase, net_flag='vgg'):
        """

        """
        super(LaneNetBinarySeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        self._decoder = FCNDecoder()
        return
    # 类对象的提示信息
    def __str__(self):
        """
        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info
    # 编码&译码
    def build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret

    #计算损失函数 返回loss值
    def compute_loss(self, input_tensor, label, name):

        """
        计算损失函数
        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算损失
            decode_logits = inference_ret['logits']
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits, labels=tf.squeeze(label, squeeze_dims=[3]),
                name='entropy_loss')
            ret = dict()
            ret['entropy_loss'] = loss
            ret['inference_logits'] = inference_ret['logits']
            return ret




