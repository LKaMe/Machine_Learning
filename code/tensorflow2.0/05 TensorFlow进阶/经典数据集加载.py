import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import detasets #导入经典数据集加载模块

#加载MNIST数据集
(x,y),(x_test,y_test) = datasets.mnist.load_data()
print('x:',x.shape,'y:',y.shape,'x test:',x_test.shape,'y test:',y_test)

train_db = tf.data.Dataset.from_tensor_slices((x,y))#构建Dataset对象


#随机打散
train_db = train_db.shuffle(10000)#随机打散样本，不会打乱样本与标签映射关系



#批训练
train_db = train_db.batch(128)#设置批训练，batch size为128



#预处理
#预处理函数实现在preprocess函数中，传入函数名即可
train_db = train_db.map(preprocess)

def preprocess(x,y):#自定义的预处理函数
    #调用此函数时会自动传入x,y对象，shape为[b,28,28],[b]
    #标准化到0~1
    x = tf.cast(x,dtype = tf.float32)/255.
    x = tf.reshape(x,[-1.28*28])#打平
    y = tf.cast(y,dtype = tf.int32)#转成整型张量
    y = tf.one_hot(y,depth = 10)#one-hot编码
    #返回的x,y将替换传入的x,y参数，从而实现数据的预处理功能
    return x,y
    
#循环训练
for step,(x,y) in enumerate(train_db):#迭代数据集对象，带step参数或
for x,y in train_db:#迭代数据集对象
for epoch in range(20):#训练Epoch数
for step,(x,y) in enumerate(train_db):

for epoch in range(20):#训练Epoch数
    for step,(x,y) in enumerate(train_db):#迭代Step数
        #train...
train_db = train_db.repeat(20)#数据集迭代20遍才终止
