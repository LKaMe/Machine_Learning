import tensorflow as tf 
import numpy as np

#向量范数
x = tf.ones([2,2])
tf.norm(x,ord = 1)#计算L1范数
tf.norm(x,ord = 2)#计算L2范数
tf.norm(x,ord = np.inf)#计算无穷范数



#最值、均值、和
x = tf.random.norma([4,10])#模型生成概率
tf.reduce_max(x,axis = 1)#统计概率维度上的最大值
tf.reduce_min(x,axis = 1)#统计概率维度上的最小值
tf.reduce_mean(x,axis = 1)#统计概率维度上的均值
x = tf.random.normal([4,10])
#统计全局的最大、最小、均值、和，返回的张量均为标量
tf.reduce_max(x),tf.reduce_min(x),tf.reduce_mean(x)
out = tf.random.normal([4,10])#模拟网络预测输出
y = tf.constant([1,2,2,0])#模拟真实标签
y = tf.one_hot(y,depth = 10)#one-hot编码

loss = keras.losses.mse(y,out)#计算每个样本的误差
loss = tf.reduce_mean(loss)#平均误差，在样本数维度上取平均值
loss#误差标量

out = tf.random.normal([4,10])
tf.reduce_sum(out,axis = -1)#求最后一个维度的和

out = tf.random.normal([2,10])
out = tf.nn.softmax(out,axis = 1)#通过softmax函数转换为概率值
out

pred = tf.argmax(out,axis = 1)#选取概率最大的位置

