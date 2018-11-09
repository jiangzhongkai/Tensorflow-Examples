"""-*- coding: utf-8 -*-
 DateTime   : 2018/11/8 14:52
 Author  : Peter_Bonnie
 FileName    : RNN.py
 Software: PyCharm
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import scipy.io.wavfile
from tensorflow.python.client import timeline
import os
import sys

'''
这里主要是利用双向LSTM来做一个手写识别的分类问题
'''
'''
数据集的导入或者加载
'''
mnist=input_data.read_data_sets("MNIST/",one_hot=True)

class Config():
    """
    配置文件的类
    """
    def __init__(self,input,timestep,batchsize,hidden_unit,hidden_unit1,learning_rate,epoch,num_class):

        self.TimeStep=timestep
        self.input=input
        self.batchsize=batchsize
        self.hidden_unit=hidden_unit
        self.hidden_unit1=hidden_unit1
        self.learning_rate=learning_rate
        self.epoch=epoch
        self.num_class=num_class
        self.weight ={
            'in':tf.Variable(tf.random_normal([2 * self.hidden_unit1, self.hidden_unit])),
            'out':tf.Variable(tf.random_normal([2*self.hidden_unit,self.num_class]))
        }
        self.bias = {
            'in':tf.Variable(tf.random_normal([self.hidden_unit])),
            'out':tf.Variable(tf.random_normal([self.num_class]))
        }
        self.max_samples=400000


class Data_Processing():
    """
    数据处理的类
    """
    def __init__(self):

        self._TRAIN_PATH=''
        self._TEST_PATH=''

    def data_processing(self):
        """
        数据的读取与加载
        :return:
        """
        pass

def BiLSTM_Model(x,config):
    """
    双向LSTM模型来对图像进行分类
    :return:
    """
    '''
    LSTM在进行对序列数据进行处理的时候，需要先将其转化为满足网络的格式[batch，timestep,features]
    '''
    x=tf.transpose(x,[1,0,2])
    x=tf.reshape(x,[-1,config.input])
    x=tf.split(x,config.TimeStep,0)

    #进行的多层双向神经网络
    fw_lstm_cell_1=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_unit1)
    bw_lstm_cell_1=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_unit1)

    fw_lstm_cell_2=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_unit)
    bw_lstm_cell_2=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_unit)


    stack_lstm_fw=tf.nn.rnn_cell.MultiRNNCell(cells=[fw_lstm_cell_1,fw_lstm_cell_2])
    stack_lstm_bw=tf.nn.rnn_cell.MultiRNNCell(cells=[bw_lstm_cell_1,bw_lstm_cell_2])

    outputs,_,_=tf.nn.static_bidirectional_rnn(cell_fw=stack_lstm_fw,cell_bw=stack_lstm_bw,inputs=x,dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1],config.weight['out']),config.bias['out'])  #全连接层进行输出

if __name__=="__main__":

   #定义一个配置类的对象
   config=Config(learning_rate=0.01,batchsize=128,input=28,timestep=28,hidden_unit1=256,num_class=10,epoch=None,hidden_unit=128)

   #定义变量和占位符
   #None 表示不确定一次输入多少条数据
   X=tf.placeholder(dtype=tf.float32,shape=[None,config.TimeStep,config.input])
   Y=tf.placeholder(dtype=tf.float32,shape=[None,config.num_class])

   #预测结果的输出
   pred=BiLSTM_Model(X,config)

   #计算交叉熵损失
   cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=pred))
   optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

   #计算预测的准确性
   correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
   accuracy=tf.reduce_sum(tf.cast(correct_pred,tf.float32))

   #接下来开始对变量进行初始化
   init=tf.global_variables_initializer()
   pred_result=[]
   with tf.Session() as sess:
       sess.run(init)
       step = 1
       while step * config.batchsize <config.max_samples:
           batch_x, batch_y = mnist.train.next_batch(config.batchsize)
           batch_x = batch_x.reshape((config.batchsize, config.TimeStep, config.input))
           sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
           if step%20==0:
               acc,loss,pred_,pre_ = sess.run([accuracy/config.batchsize, cost,pred,pre], feed_dict={X: batch_x, Y: batch_y})
               pred_result.append(sess.run(tf.argmax(pred_,1)))  #将预测的结果bao
               print(sess.run(pre_))
               print("acc={:.5f},loss={:.9f},prediction:{}".format(acc, loss,pred_result[-1]))

           step+=1

       print("Optimizer Finished!!!")

       test_len=10000
       test_data=mnist.test.images[:test_len].reshape((-1,config.TimeStep,config.input))
       test_label=mnist.test.labels[:test_len]
       print("Testing Accuracy:{:.5f}".format(sess.run(accuracy/test_len,feed_dict={X:test_data,Y:test_label})))







