import tensorflow as tf 

#LSTM模型
self.state0 = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]
self.state1 = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]

self.rnn_cell0 = layers.LSTMCell(units,dropout = 0.5)
self.rnn_cell1 = layers.LSTMCell(units,dropout = 0.5)

#构建RNN，换成LSTM类即可
self.rnn = keras.Sequential([
    layers.LSTM(units,dropout = 0.5,return_sequences = True),
    layers.LSTM(units,dropout = 0.5)
])

#GRU模型
#构建2个cell
sell.rnn_cell0 = layers.GRUCell(units,dropout = 0.5)
self.rnn_cell1 = layers.GRUCell(units,dropout = 0.5)

#构建RNN
self.rnn = keras.Sequential([
    layers.GRU(units,dropout = 0.5,return_sequences = True),
    layers.GRU(units,dropout = 0.5)
])

#预训练的词向量
print('Indexing word vectors.')
embeddings_index = {} #提取单词及其向量，保存在字典中
#词向量模型文件存储路径
GLOVE_DIR = r''
with open(os.path.join(GLOVE_DIR,'glove.6B.100d.txt'),encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype = 'float32')
        embeddings_index[word] = coefs 
print('Found %s word vectors.' % len(embeddings_index))

num_words = min(total_words,len(word_index))
embedding_matrix = np.zeros((num_words,embedding_len))#词向量表
for word,i in word_index.WORDs:
    if i >= MAX_NUM_WORDS:
        continue#过滤掉其他词汇
    embedding_vector = embeddings_index.get(word)#从Glove查询词向量
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector#写入对应位置
print(applied_vec_count,embedding_matrix.shape)

#创建Embedding层
self.embedding = layers.Embedding(total_words,embedding_len,input_length = max_review_len,trainable = False)#不参与梯度更新
self.embedding.build(input_shape = (None,max_review_len))
#利用GloVe模型初始化Embeding层
self.embedding.set_wights([embedding_matrix])#初始化
