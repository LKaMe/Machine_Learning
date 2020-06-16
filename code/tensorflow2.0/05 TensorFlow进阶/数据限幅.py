import tensorflow as tf
x = tf.range(9)
tf.maximum(x,2)#下限幅到2
tf.minimum(x,7)#上限幅到7
#基于tf.maximum函数，实现ReLU函数
def relu(x):#ReLU函数
    return tf.maximum(x,0.)#下限幅为0即可

x = tf.range(9)
tf.minimum(tf.maximum(x,2),7)#限幅为2~7
