#张量方式实现
#隐藏层1张量
w1 = tf.Variable(tf.random.truncated_normal([784,256],steddev = 0.1))
b1 = tf.Variable(tf.zeros([256]))
#隐藏层2张量
w2 = tf.Variable(tf.random.truncated_normal([256,128],steddev = 0.1))
b2 = tf.Variable(tf.zeros([128]))
#隐藏层3张量
w3 = tf.Variable(tf.random.truncated_normal([128,64],steddev = o.1))
b3 = tf.Variable(tf.zeroos([64]))
#输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64,10],steddev = 0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:#梯度记录器
    #x:[b,28*28]
    #隐藏层1前向计算，[b,28*28] =>[b,256]
    h1 = x @ w1 + tf.broadcast_to(b1,[x.shape[0],256])
    h1 = tf.nn.relu(h1)
    #隐藏层2前向计算，[b,256]=>[b,128]
    h2 = h1 @ w2 + b2 
    h2 = tf.nn.relu(h2)
    #隐藏层3前向计算，[b,128] => [b,64]
    h3 = h2 @ w3 + b3
    h3 = tf.nn.relu(h3)
    #输出层前向计算，[b,64] => [b,10]
    h4 = h3 @ w4 + b4




#层方式实现
#导入常用网络层layers
from tensorflow.keras import layers.Sequential

fc1 = layers.Dense(256,activation = tf.nn.relu)#隐藏层1
fc2 = layers.Dense(128,activation = tf.nn.relu)#隐藏层2
fc3 = layers.Dense(64,activation = tf.nn.relu)#隐藏层3
fc4 = layers.Dense(10,activation = None)#输出层
#在前向计算时，依序通过各个网络层即可
x = tf.random.normal([4,28*28])
h1 = fc1(x)#通过隐藏层1得到输出
h2 = fc2(h1)#通过隐藏层2得到输出
h3 = fc3(h2)#通过隐藏层3得到输出
h4 = fc4(h3)#通过输出层得到网络输出


#对于这种数据依次向前传播的网络，也可以通过Sequential容器封装成一个网络大类对象，调用大类的前向计算函数一次即可完成所有层的前向计算
#导入Sequential容器
from tensorflow.keras import layers,Sequential 

#通过Sequential容器封装为一个网络类
model = Sequential([
    layers.Dense(256,activation = tf.nn.relu),#创建隐藏层1
    layers.Dense(128,activation = tf.nn.relu),#创建隐藏层2
    layers.Dense(64,activation = tf.nn.relu),#创建隐藏层3
    layers.Dense(10,activation = None),#创建输出层
])
out = model(x)#前向计算得到输出
