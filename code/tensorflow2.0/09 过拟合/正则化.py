import tensorflow as tf 

#L1正则化
#创建网络参数w1,w2
w1 = tf.random.normal([4,3])
w2 = tf.random.normal([4,2])
#计算L1正则化项
loss_reg = tf.reduce_sum(tf.math.abs(w1))\+tf.reduce_sum(tf.math.abs(w2))

#L2正则化
#创建网络参数w1,w2
w1 = tf.random.normal([4,3])
w2 = tf.random.normal([4,2])
#计算L2正则化项
loss_reg = tf.reduce_sum(tf.square(w1))\+tf.reduce_sum(tf.square(w2))

