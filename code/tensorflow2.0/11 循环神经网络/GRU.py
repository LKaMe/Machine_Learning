import tensorflow as tf 
#初始化状态向量，GRU只有一个
h = [tf.zeros([2,64])]
cell = layers.GRUCell(64)#新建GRU Cell,向量长度为64
#在时间戳维度上解开，循环通过cell
for xt in tf.unstack(x,axis = 1):
    out,h = cell(xt,h)
#输出形状
out.shape 

net = keras.Sequential([
    layers.GRU(64,return_sequences=True),
    layers.GRU(64)
])
out = net(x)

