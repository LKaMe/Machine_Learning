from keras.layers import Embedding

Embedding_layer = Embedding(1000,64)#Embedding层至少需要两个参数：标记的个数(这里是1000，即最大单词索引+1)和嵌入的维度(这里是64)
