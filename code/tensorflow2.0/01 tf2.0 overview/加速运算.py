import tensorflow as tf 
#矩阵A的shape为[1,n],矩阵B的shape为[n,1],通过调节n可控制矩阵的大小

#创建在CPU环境上运算的2个矩阵
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1,n])
    cpu_b = tf.random.normal([n,1])
    print(cpu_a.device,cpu_b.device)

#创建使用GPU环境运算的2个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1,n])
    gpu_b = tf.random.normal([n,1])
    print(gpu_a.device,gpu_b.device)

def cpu_run():#CPU运算函数
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a,cpu_b)
    return c

def gpu_run():#gpu运算函数
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a,gpu_b)
    return c

