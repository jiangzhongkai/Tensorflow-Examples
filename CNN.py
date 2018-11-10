"""
coding:utf-8
@author:Peter
@datetime:2018-11-10
"""
'''
主要是在CIFAR10数据集上的图像识别
'''
"""
主要层：
   conv->relu->maxpool->...->conv->relu->maxpool->flatten->fc->fc->output  分类或者预测
"""
import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib

sess=tf.Session()

class Config():
    """
    模型参数配置
    """
    def __init__(self):
        self.batch_size=128
        self.output_every=50
        self.generations=20000
        self.num_channels=3
        self.eval_every=500
        self.image_height=32
        self.image_width=32
        self.crop_height=24
        self.crop_width=24
        self.num_classes=10
        self.data_dir='temp'
        self.extract_folder="cifar_10"
        self.learning_rate=0.1
        self.lr_decay=0.9
        self.num_gens_to_wait=250
        self.image_vec_length=self.image_height*self.image_width*self.num_channels
        self.record_length=1+self.image_vec_length

'''设置下载数据集的url和存放地址'''
data_dir="temp"

def load_dataset(data_dir):
    '''主要是用来加载数据'''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pass

def read_files(filename_queue,distort_images=True,config=Config()):
    '''1.建立图片读取器，然后返回一个随机打乱的图片
       2.首先声明一个读取固定长度的读取器
       3.从图像队列中读取图片，抽取图片并标记
       4.修改这个图片并返回随机打乱的这个图片
    '''
    reader=tf.FixedLengthRecordReader(record_bytes=config.record_length)
    key,record_string=reader.read(filename_queue)
    record_bytes=tf.decode_raw(record_string,tf.uint8)

    '''提取图片的标签'''
    image_label=tf.cast(tf.slice(record_bytes,[0],[1]),tf.int32)
    '''提取图片'''
    images=tf.reshape(tf.slice(record_bytes,[1],[config.image_vec_length]),[config.num_channels,config.image_height,config.image_width])
    '''改变图片的形状'''
    images=tf.transpose(images,[1,2,0])
    '''这里应该是其像素点的转换'''
    reshape_image=tf.cast(images,tf.float32)
    '''对图片进行随机裁剪'''
    after_crop_image=tf.image.resize_image_with_crop_or_pad(reshape_image,config.crop_height,config.crop_width)
    '''对原始图像进行翻转、对比度、亮度调整'''
    if distort_images:
        '''进行随机翻转'''
        final_image=tf.image.random_flip_left_right(after_crop_image)
        '''对亮度进行调整'''
        final_image=tf.image.random_brightness(final_image,max_delta=63)
        '''对比度进行调整'''
        final_image=tf.image.random_contrast(final_image,lower=0.5,upper=2.0)

        final_image=tf.image.per_image_standardization(final_image)

    return (final_image,image_label)








