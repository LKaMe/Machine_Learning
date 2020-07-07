import tensorflow as tf 

x = tf.random.normal([2,80,100])
xt = x[:,0,:]#得到一个时间戳的输入
cell = layers.LSTM.LSTMCell(64)#创建LSTM Cell 
#初始化状态和输出List,[h,c]
state = [tf.zeros([2,64]),tf.zeros([2,64])]
out,state = cell(xt,state) #前向计算

#在序列长度维度上解开，循环送入LSTM Cell单元
for xt in tf.unstack(x,axis = 1):
    #前向计算
    out,state = cell(xt,state)

#LSTM层
#创建一层LSTM层，内存向量长度为64 
layer = layers.LSTM(64)
#序列通过LSTM层，默认返回最后一个时间戳的输出 h
out = layer(x)

#创建LSTM层时，设置返回每个时间戳上的输出
layer = layers.LSTM(64,return_sequences=True)
#前向计算，每个时间戳上的输出自动进行了concat,拼成一个张量
out = layer(x)

#非末层的LSTM层需要上一层在所有的时间戳的输出作为输入
#和CNN网络一样，LSTM也可以简单地层层堆叠
net = keras.Sequential([
    layers.LSTM(64,return_sequences=True),#非末层需要返回所有时间戳输出
    layers.LSTM(64)
])
#一次通过网络模型，即可得到最末层、最后一个时间戳的输出
out = net(x)
