from keras.datasets import imdb 
from keras.preprocessing import sequence 

max_features = 10000 #作为特征的单词个数
maxlen = 500 #在这么多单词之后截断文本(这些单词都属于前max_features个最常见的单词)
batch_size = 32

print('Loading data...')
(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words = max_features)
print(len(input_train),'train sequences')
print(len(input_test),'test sequences')

print('Pad sequences(sample x time)')
input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)
print('input_train shape:',input_train.shape)
print('input_test shape:',input_test.shape)