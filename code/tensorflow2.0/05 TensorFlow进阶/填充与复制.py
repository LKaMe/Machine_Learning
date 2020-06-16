import tensorflow as tf
#填充

a = tf.constant([1,2,3,4,5,6])#第一个句子
b = tf.constant([7,8,1,6])#第二个句子
b = tf.pad(b,[[0,2]])#句子末尾填充2个0
b#填充后的结果

tf.stack([a,b],axis = 0)#堆叠合并，创建句子数维度

total_words = 10000 #设定词汇量大小
max_review_len = 80 #最大句子长度
embedding_len = 100 #词向量长度
#加载IMDB数据集
(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words = total_words)
#将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen = max_review_len,truncating = 'post',padding = 'post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen = max_review_len,truncating = 'post',padding = 'post')
print(x_train.shape,x_test.shape)#打印等长的句子张量形状
x = tf.random.normal([4,28,28,1])
#图片上下、左右各填充2个单元
tf.pad(x,[[0,0],[2,2],[2,2],[0,0]])