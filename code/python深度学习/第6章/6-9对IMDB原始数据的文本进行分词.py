# from keras.preprocessing.text import Tokenizer 
# from keras.preprocessing.sequence import pad_sequences 
# import numpy as np 

# maxlen = 100 #在100个单词后截断评论
# training_samples = 100 #在200个样本上训练
# validation_samples = 10000 #在10000个样本上验证
# max_words = 10000 #只考虑数据集中前10000个最常见的单词

# tokenizer = Tokenizer(num_words = max_words)
# tokenizer.fit_on_text(texts)
# sequences = tokenizer.texts_to_sequences(texts)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

# data = pad_sequences(sequences,maxlen = maxlen)

# labels = np.asarray(labels)
# print('Shape of data tensor:',data.shape)
# print('Shape of label tensor:',labels.shape)

# indices = np.arange(data.shape[0])#将数据集划分为训练集和验证集，但首先要打乱数据，因为一开始数据中的样本是排好序的(所有负面评论都在前面，然后是所有正面评论)
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]

# x_train = data[:training_samples]
# y_train = labels[:training_samples]
# x_val = data[training_samples:training_samples + validation_samples]
# y_val = labels[training_samples:training_samples + validation_samples]

