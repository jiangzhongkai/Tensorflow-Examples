"""
coding:utf-8
@author:Peter
@datetime:2018-11-10
"""
'''
主要是在CIFAR10数据集上的图像识别
'''
"""
主要结构：
   conv->relu->maxpool->batchnorm->conv->relu->maxpool->batchnorm->flatten->fc->fc->output 
   
   
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


def cifar_cnn_model(input_images,batch_size,train_logical=True):
    '''搭建图像分类的模型'''
    '''initital variables'''
    def truncated_normal_var(name,shape,dtype):
        return (tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.truncated_normal_initializer(stddev=0.05)))

    def zero_var(name,shape,dtype):
        return (tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.truncated_normal_initializer(0.0)))

    '''first conv-layer'''

    '''卷积层->激活层->池化层->归一化层'''

    with tf.variable_scope("conv1") as scope:
        first_conv_kernel=truncated_normal_var(name="conv_kernel",shape=[5,5,3,64],dtype=tf.float32)  #滤波器

        '''卷积层'''
        conv1=tf.layers.conv2d(input_images,first_conv_kernel,[1,1,1,1],padding="SAME")
        conv1_bias=zero_var(name="conv1_bias",shape=[64],dtype=tf.float32)
        conv1_add_bias=tf.nn.bias_add(conv1,conv1_bias)
        relu_conv1=tf.nn.relu(conv1_add_bias)
        '''池化层'''
        pool1=tf.nn.max_pool(relu_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool_layer_1")
        normal=tf.nn.lrn(pool1,bias=2.0)

    with tf.variable_scope("conv2") as scope:
        second_conv_kernel=truncated_normal_var(name="conv_kernel_2",shape=[5,5,64,64],dtype=tf.float32)   #滤波器

        conv2=tf.layers.conv2d(normal,second_conv_kernel,[1,1,1,1],padding="SAME")
        conv2_bias=zero_var(name="conv2_bias",shape=[64],dtype=tf.float32)
        conv2_add_bias=tf.nn.bias_add(conv2,conv2_bias)
        relu_conv2=tf.nn.relu(conv2_add_bias)
        '''池化层'''
        pool2=tf.nn.max_pool(relu_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool_layer_2")

        normal_2=tf.nn.lrn(pool2,depth_radius=5,bias=2.0)
        reshaped_output=tf.reshape(normal_2,shape=[batch_size,-1])
        reshaped_dim=reshaped_output.get_shape()[1].value

    with tf.variable_scope("fully_connected_1") as scope:

        '''初始化全连接层的权重和偏置'''
        fully_weight_1=truncated_normal_var(name="fully_weight_1",shape=[reshaped_dim,384],dtype=tf.float32)

        fully_bias_1=zero_var(name="fully_bias_1",shape=[384],dtype=tf.float32)

        fully_layer_1=tf.nn.relu(tf.matmul(reshaped_output,fully_weight_1)+fully_bias_1)

    with tf.variable_scope("fully_connected_2") as scope:

        '''初始化权重和偏置'''
        fully_weight_2=truncated_normal_var(name="fully_weight_2",shape=[384,192],dtype=tf.float32)

        fully_bias_2=zero_var(name="fully_bias_2",shape=[192],dtype=tf.float32)

        fully_layer_2=tf.nn.relu(tf.matmul(fully_layer_1,fully_weight_2)+fully_bias_2)

    with tf.variable_scope("final_layer") as scope:

        final_layer_weight=truncated_normal_var(name="final_layer_weight",shape=[192,10],dtype=tf.float32)

        final_layer_bias=zero_var(name="final_layer_bias",shape=[10],dtype=tf.float32)

        final_layer_output=tf.nn.relu(tf.matmul(fully_layer_2,final_layer_weight)+final_layer_bias)

    return final_layer_output   #shape=[batch_size,num_classes]

def create_loss(pred,true):
    '''创建损失函数'''
    true=tf.squeeze(tf.cast(true,tf.int32))  #主要是类型转换以及压缩维度
    cross_entry=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=true)
    final_entry=tf.reduce_mean(cross_entry)
    return final_entry



if __name__=="__main__":
    print(cifar_cnn_model())






