from keras.datasets import imdb 
from keras.layers import preprocessing 
max_features = 10000#作为特征的单词个数
maxlen = 20#在这么多单词后截断文本(这些单词都属于前max_features个最常见的单词)
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)#将数据加载为整数列表

x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)#将整数列表转换成形状为(samples,maxlen)的二维整数张量
x_test= preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
