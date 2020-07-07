import tensorflow as tf 
cell = layers.SimpleRNNCell(3)#创建RNN Cell,内存向量长度为3
cell.build(input_shape=(None,4))#输出特征长度n=4
print(cell.trainable_variable)#打印wxh,whh,b张量

#初始化状态向量，用列表包裹，统一格式
h0 = [tf.zeros([4,64])]
x = tf.random.normal([4,80,100])#生成输入张量，4个80单词的句子
xt = x[:,0,:]#所有句子的第一个单词
#构建输入特征n = 100,序列长度s = 80,状态长度=64的Cell
cell = layers.SimpleRNNCell(64)
out,h1 = cell(xt,h0)#前向计算
print(out.shape,h1[0].shape)
h = h0 #h保存每个时间戳上的状态向量列表
#在序列长度的维度解开输入，得到xt:[b,n]
for xt in tf.unstack(x,axis = 1):
    out,h = cell(xt,h)#前向计算，out和h均被覆盖
#最终输出可以聚合每个时间戳上的输出，也可以只取最后时间戳的输出
out = out 


#多层SimpleRNNCell网络
x = tf.random.normal([4,80,100])
xt = x[:,0,:]#取第一个时间戳的输入x0
#构建2个Cell,先cell0,后cell1,内存状态向量长度都为64
cell0 = layers.SimpleRNNCell(64)
cell1 = layers.SimpleRNNCell(64)
h0 = [tf.zeros([4,64])]#cell0的初始状态向量
h1 = [tf.zeros([4,64])]#cell1的初始状态向量
for xt in tf.unstack(x,axis = 1):
    #xt作为输入，输出为out0
    out0,h0 = cell0(xt,h0)
    #上一个cell的输出out0作为本cell的输入
    out1,h1 = cell1(out0,h1)

#保存上一层的所有时间戳上面的输出
middle_sequences = []
#计算第一层的所有时间戳上的输出，并保存
for xt in tf.unstack(x,axis = 1):
    out0,h0 = cell0(xt,h0)
    middle_sequence.append(out0)
#计算第二层的所有时间戳上的输出
#如果并不是末层，需要保存所有时间戳上面的输出
for xt in middle_sequences:
    out1,h1 = cell1(xt,h1)

layer = layers.SimpleRNN(64)#创建状态向量长度为64的SimpleRNN层
x = tf.random.normal([4,80,100])
out = layer(x)#和普通卷积网络一样，一行代码即可获得输出
print(out.shape)

#创建RNN层时，设置返回所有时间戳上的输出
layer = layers.SimpleRNN(64,return_sequences=True)
out = layer(x)#前向计算
print(out)#输出，自动进行了concat操作

net = keras.Sequential([#构建2层RNN网络
#除最末层外，都需要返回所有时间戳的输出，用作下一层的输入
layers.SimpleRNN(64,return_sequences = True),
layers.SimpleRNN(64),
])

out = net(x)#前向计算
