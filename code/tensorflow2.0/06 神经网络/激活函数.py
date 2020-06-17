import tensorflow as tf

#Sigmoid函数
x = tf.linspace(-6,6.,10)#构造-6~6的输入向量
tf.nn.sigmoid(x)#通过Sigmoid函数


#ReLU函数
tf.nn.relu(x)#通过ReLU激活函数


#LeakyReLU函数
tf.nn.leaky_relu(x,alpha = 0.1)#通过LeakyReLU激活函数



#Tanh函数
tf.nn.tanh(x)#通过tanh激活函数


