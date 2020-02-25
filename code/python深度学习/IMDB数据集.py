# from keras.datasets import imdb

# #加载IMDB数据集，仅保留训练数据中前10000个最常出现的单词，低频单词将被舍弃
# (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# print(train_data[0])
# print(train_labels[0])
# print(max([max(sequence)for sequence in train_data]))
# word_index = imdb.get_word_index()
# revere_word_index = dict([(value,key) foor (key,value) in word_index.items()])#键值颠倒，将整数索引映射为单词
# decoded_review = ' '.join([reverse_word_index.get(i - 3 , '?') for i in train_data[0]])
# #将评论解码。注意，索引减去了3，因为0，1，2是为'padding'(填充)、'start of sequence'(序列开始)、'unknown'(未知词)分别保留的索引

 