import tensorflow as tf 

#Embedding层
x = tf.range(10)#生成10个单词的数字编码
x = tf.random.shuffle(x)#打散
#创建共10个单词，每个单词用长度为4的向量表示的层
net = layers.Embedding(10,4)
out = net(x)

#从预训练模型中加载词向量表
embed_glove = load_embed('glove.6B.50d.txt')
#直接利用预训练的词向量表初始化Embedding层
net.set_weights([embed_glove])

