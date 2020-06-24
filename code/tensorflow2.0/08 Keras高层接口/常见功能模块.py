# #常见网络层类
# import tensorflow as tf 
# #导入Keras模块，不能使用import keras,它导入的是标准的Keras库
# from tensorflow import keras 
# from tensorflow.keras import layers #导入常见网络层类
# #然后创建Softmax层，并调用__call__方法完成前向计算
# x = tf.constant([2.,1.,0.1])#创建输入张量
# layer = layers.Softmax(axis = -1)#创建Softmax层
# out = layer(x)#调用softmax前向计算，输出为out
# #经过Softmax网络层后，得到概率分布out
# out = tf.nn.softmax(x)#调用softmax函数完成前向计算

#网络容器
#例如，2层的全连接层加上单独的激活函数层，可以通过Sequential容器封装为一个网络
#导入Sequential容器
from tensorflow.keras import layers,Sequential 
network = Sequential([ #封装为一个网络
    layers.Dense(3,activation=None),#全连接层，此处不使用激活函数
    layers.ReLU(),#激活函数层
    layers.Dense(2,activation=None),#全连接层，此处不使用激活函数
    layers.ReLU()#激活函数层
])
x = tf.random.normal([4,3])
out = network(x) #输入从第一层开始，逐层传播至输出层，并返回输出层的输出
#Sequential容器也可以通过add()方法继续追加新的网络层，实现动态创建网络的功能
layers_num = 2#堆叠2次
network = Sequential([])#先创建空的网络容器
for _in range(layers_num):
    network.add(layers.Dense(3))#添加全连接层
    network.add(layers.ReLU())#添加激活函数层
network.build(input_shape=(4,4))#创建网络参数
network.summary()

#打印网络的待优化参数名与shape
for p in network.trainable_variables:
    print(p.name,p.shape)#参数名和形状
    