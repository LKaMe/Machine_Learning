import tensorflow as tf 

#常见的误差函数有均方差、交叉熵、KL散度、Hinge Loss函数等
#其中均方差函数和交叉熵函数在深度学习中比较常见，均方差函数主要用于回归问题，交叉熵函数主要用于分类问题

#均方差误差函数
o = tf.random.normal([2,10])#构造网络输出
y_onehot = tf.constant([1,3])#构造真实值
y_onehot = tf.one_hot(y_onehot,depth = 10)
loss = keras.losses.MSE(y_onehot,o)#计算均方差
loss = tf.reduce_mean(loss)#计算batch均方差
#创建MSE类
criteon = keras.losses.MeanSquaredError()
loss = criteon(y_onehot,o)#计算batch均方差


#交叉熵误差函数

