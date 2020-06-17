#张量方式实现
#创建w,b张量
x = tf.random.normal([2,784])
w1 = tf.Variable(tf.random.truncated_normal([784,256],steddev = 0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1) + b1 #线性变换
o1 = tf.nn.relu(o1)#激活函数


#层方式实现
x = tf.random.normal([4,28*28])
from tensorflow.keras import layers #导入层模块
#创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(512,activation = tf.nn.relu)
h1 = fc(x)#通过fc实例完成一次全连接层的计算，返回输出张量
fc.kernel#获取Dense类的权值矩阵
fc.bias#获取Dense类的偏置向量
fc.trainable_variables#返回待优化参数列表
fc.variables#返回所有参数列表


