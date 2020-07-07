import tensorflow as tf 
batchsz = 128 #批量大小
total_words = 10000 #词汇表大小N_vocab 
max_review_len = 80 #句子最大长度s，大于的句子部分将截断，小于的将填充
embedding_len = 100 #词向量特征长度n
#加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=total_words)
#打印输入的形状，标签的形状
print(x_train.shape,len(x_train[0]),y_train.shape)
print(x_test.shape,len(x_test[0]),y_test.shape)


#数字编码表
word_index = keras.datasets.imdb.get_word_index()
#打印出编码表的单词和对应的数字
for x,v in word_index.items():
    print(k,v)

#前面4个ID是特殊位
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0 #填充标志
word_index["<START>"] = 1 #起始标志
word_index["<UNK>"] = 2 #未知单词的标志
word_index["<UNUSED>"] = 3
#翻转编码表
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

#截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen = max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen = max_review_len)

#构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch 
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(batchsz,drop_remainder=True)
db_test = db_test.batch(batchsz,drop_remainder=True)
#统计数据集属性
print('x_train shape:',x_train.shape,tf.reduce_max(y_train),tf.reduce_min(y_train))
print('x_test shape:',x_test.shape)

class MyRNN(keras.Model):
    #cell方式构建多层网络
    def __init__(self,units):
        super(MyRNN,self).__init__()
        #[b,64],构建cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz,units])]
        self.state1 = [tf.zeros([batchsz,units])]
        #词向量编码[b,80] => [b,80,100]
        self.embedding = layers.Embedding(total_words,embedding_len,input_length = max_review_len)
        #构建2个Cell,使用dropout技术防止过拟合
        self.rnn_cell0 = layers.SimpleRNNCell(units,dropout = 0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units,dropout = 0.5)
        #构建分类网络，用于将CELL的输出特征进行分类，2分类
        #[b,80,100] => [b,64] => [b,1]
        self.outlayer = layers.Dense(1)

    def call(self,inputs,training = None):
        x = inputs #[b,80]
        #获取词向量：[b,80] => [b,80,100]
        x = self.embedding(x)
        #通过2个RNN CELL，[b,80,100] => [b,64]
        state0 = self.state0 
        state1 = self.state1 
        for word in tf.unstack(x,axis = 1):#word:[b,100]
            out0,state0 = self.rnn_cell0(word,state0,training)
            out1,state1 = self.rnn_cell1(out0,state1,training)
        #末层最后一个输出作为分类网络的输入:[b,64] => [b,1]
        x = self.outlayer(out1,training)
        #通过激活函数，p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob 


#训练与测试
def main():
    units = 64#RNN状态向量长度n
    epochs = 20 #训练epochs 

    model = MyRNN(units)#创建模型
    #装配
    model.compile(optimizer = optimizer.Adam(0.001),loss = losses.BinaryCrossentropy(),metrics = ['accuracy'])
    #训练和验证
    model.fit(db_train,epochs = epochs,validation_data = db_test)
    #测试
    model.evaluate(db_test)
W = tf.ones([2,2])#任意创建某矩阵
eigenvalues = tf.linalg.eigh(w)[0] #计算矩阵的特征值

val = [W]
for i in range(10):#矩阵相乘n次方
    val.append([val[-1]@W])
#计算L2范数
norm = list(map(lambda x : tf.norm(x).norm(),val))

W = tf.ones([2,2])*0.4 #任意创建某矩阵 
eigenvalues = tf.linalg.eigh(W)[0] #计算特征值
print(eigenvalues)

val = [W]
for i in range(10):
    val.append([val[-1]@W])
#计算L2范数
norm = list(map(lambda x:tf.norm(x).numpy(),val))
plt.plot(range(1,12),norm)

#梯度剪裁
a = tf.random.uniform([2,2])
tf.clip_by_value(a,0.4,0.6)#梯度值剪裁
a = tf.random.uniform([2,2])*5 
#按范数方式剪裁
b = tf.clip_by_norm(a,5)
#剪裁前和剪裁后的张量范数
tf.norm(a),tf.norm(b)

w1 = tf.random.normal([3,3])#创建梯度张量1
w2 = tf.random.normal([3,3])#创建梯度张量2
#计算global norm 
global_norm = tf.math.sqrt(tf.norm(w1) ** 2 + tf.norm(w2) ** 2)
#根据global norm和max norm=2剪裁
(ww1,ww2),global_norm = tf.clip_by_global_norm([w1,w2],2)
#计算剪裁后的张量组的global norm
global_norm2 = tf.math.sqrt(tf.norm(ww1) **2 + tf.norm(ww2) ** 2)
#打印裁剪前的全局范数和裁剪后的全局范数
print(global_norm,global_norm2)

#在网络训练时，梯度裁剪一般在计算出梯度后，梯度更新之前进行
with tf.GradientTape() as tape:
    logits = model(x) #前向传播
    loss = criteon(y,logits)#误差计算
#计算梯度值
grads = tape.gradient(loss,model.trainable_variables)
grads,_=tf.clip_by_global_norm(grads,25)#全局梯度裁剪
#利用裁剪后的梯度张量更新参数
optimizer.apply_gradients(zip(grads,model.trainable_variables))

