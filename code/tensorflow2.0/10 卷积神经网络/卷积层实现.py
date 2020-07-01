import tensorflow as tf 

#自定义权值
x = tf.random.normal([2,5,5,3])#模拟输入，3通道，高宽为5
##需要根据[k,k,cin,cout]格式创建w张量，4个3*3大小卷积核
w = tf.random.normal([3,3,3,4])
#步长为1，padding为0
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]])
TensorShape([2,3,3,4])
padding = [[0,0],[上，下],[左,右],[0,0]]
x = tf.random.normal([2,5,5,3])#模拟输入，3通道，高宽为5
#需要根据[k,k,cin,cout]格式创建,4个3*3大小卷积核
w = tf.random.normal([3,3,3,4])
#步长为1，padding为1
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[1,1],[1,1],[0,0]])
TensorShape([2,5,5,4])

x = tf.random.normal([2,5,5,3])#模拟输入，3通道，高宽为5
w = tf.random.normal([3,3,3,4])#4个3*3大小的卷积核
#步长为，padding设置为输出、输入同大小
#需要注意的是，padding=same只有在strides=1时才是同大小
out = tf.nn.conv2d(x,w,strides=1,padding='SAME')

x = tf.random.normal([2,5,5,3])
w = tf.random.normal([3,3,3,4])
#高宽先padding成可以整除3的最小整数6，然后6按3倍减少，得到2*2
out = tf.nn.conv2d(x,w,strides=3,padding='SAME')
#根据[cout]格式创建偏置向量
b = tf.zeros([4])
#在卷积输出上叠加偏置向量，它会自动broadcasting为[b,h',w',cout]
out = out + b 