import tensorflow as tf 

z = tf.constant([2.,1.,0.1])
tf.nn.softmax(z)#通过Softmax函数


z = tf.random.normal([2,10])#构造输出层的输出
y_onehot = tf.constant([1,3])#构造真实值
y_onehot = tf.one_hot(y_onehot,depth = 10)#one-hot编码
#输出层未使用Softmax函数，故from_logits设置为True
#这样categorical_crossentropy函数在计算损失函数前，会先内部调用Softmax函数
loss = keras.losses.categorical_crossentropy(y_onehot,z,from_logits = True)
loss = tf.reduce_mean(loss)#计算平均交叉熵损失
loss


#创建Softmax与交叉熵计算类，输出层的输出z未使用softmax
criteon = keras.losses.CategoricalCrossentropy(from_logits = True)
loss = criteon(y_onehot,z)#计算损失
loss 


#如果希望输出值的范围分布在(-1,1)区间，可以简单地使用tanh激活函数，实现如下：
x = tf.linspace(-6.,6.,10)
tf.tanh(x)#tanh激活函数
