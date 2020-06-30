import tensorflow as tf 

#添加dropout操作，断开概率为0.5 
x = tf.nn.dropout(x,rate = 0.5)
#添加Dropout层，断开概率为0.5 
model.add(layers.Dropout(rate = 0.5))
