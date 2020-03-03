from keras.preprocessing.text import Tokenizer 

samples = ['The cat sat on the mat.','The dog ate my homework.']

tokenizer = Tokenizer(num_words = 1000)#创建一个分词器(tokenizer)，设置为只考虑前1000个最常见的单词
tokenizer.fit_on_texts(samples)#构建单词索引

sequences = tokenizer.texts_to_sequences(samples)#将字符串转换为整数索引组成的列表

one_hot_results = tokenizer.texts_to_matrix(samples,mode = 'binary')#也可以直接得到one-hot二进制表示。这个分词器也支持除one-hot编码外的其他向量化模式

word_index = tokenizer.word_index#找回单词索引
print('Found %s unique tokens.' % len(word_index))
