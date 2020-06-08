import tensorflow as tf 
#1.创建输入张量，并赋予初值
a = tf.constant(2.)
b = tf.constant(4.)
#2.直接计算，并打印结果
print('a+b=',a+b)