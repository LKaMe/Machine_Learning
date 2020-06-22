import tensorflow as tf 
#利用Tensorflow自动求导来求出函数在x和y的偏导数，并循环迭代更新x和y值
#参数的初始化值对优化的影响不容忽视，可以通过尝试不同的初始化值，
#检验函数优化的极小值情况
#[1.,0.],[-4,0.],[4,0.]
x = tf.constant([4.,0.])#初始化参数

for step in range(200):#循环优化200次
    with tf.GradientTape() as tape:#梯度跟踪
        tape.watch([x])#加入梯度跟踪列表
        y = himmelblau(x)#前向传播
    #反向传播
    grads = tape.gradient(y,[x])[0]
    #更新参数，0.01为学习率
    x -= 0.01 * grads 
    #打印优化的极小值
    if step % 20 == 19:
        print('step {}: x = {},f(x) = {}'
        .format(step,x.numpy(),y.numpy()))
