import tensorflow as tf 
#创建标量
a = 1.2 #python语言方式创建标量
aa = tf.constant(1.2) #TF方式创建标量
type(a),type(aa),tf.is_tensor(aa)

##通过print(x)或x可以打印出张量x的相关信息
#向量的定义须通过List容器传给tf.constant()函数
#创建一个元素的向量
a = tf.constant([1.2])#创建一个元素的向量
print(a,a.shape)

#创建3个元素的向量:
a = tf.constant([1,2,3.])#创建3个元素的向量
print(a,a.shape)

#定义矩阵
a = tf.constant([[1,2],[3,4]])
print(a,a.shape)

#定义三维张量
a = tf.constant([[[1,2],[3,4]][[5,6],[7,8]]])




#字符串类型
#创建字符串
a = tf.constant('Hello,Deep Learning.')#创建字符串
#小写化字符串
tf.strings.lower(a)


#布尔类型张量
#创建布尔类型标量
a = tf.constant(True)
#创建布尔类型向量
a = tf.constant([True,False])
#创建TF布尔张量
a = tf.constant(True)
a is True #TF布尔类型张量与python布尔类型比较
a == True #仅数值比较


#创建指定精度的张量
tf.constant(123456789,dtype = tf.int16)
tf.constant(123456789,dtype = tf.int32)


#读取精度
print('before:',a.dtype)#读取原有张量的数值精度
if a.dtype != tf.float32:#如果精度不符合要求，则进行转换
    a = tf.cast(a,tf.float32)#tf.cast函数可以完成精度转换
print('after:',a.dtype)#打印转换后的精度


#类型转换
a = tf.constant(np.pi,dtype = tf.float16)#创建tf.float16低精度张量
tf.cast(a,tf.double)#转换为高精度张量

a = tf.constant(123456789.dtype = tf.int32)
tf.cast(a,tf.int32)#转换为低精度整型

a = tf.constant([True,False])
tf.cast(a,tf.int32)#布尔类型转整型

a = tf.constant([-1,0,1,2])
tf.cast(a,tf.bool)#整型转布尔类型



#待优化张量
a = tf.constant([-1,0,1,2])#创建TF张量
aa = tf.Variable(a) #转换为Variable类型
aa.name,aa.trainable #类型张量的属性


a = tf.Variable([1,2],[3,4])#直接创建Variable张量



#从数组、列表对象创建
tf.convert_to_tensor([1,2.])#从列表创建张量
tf.convert_to_tensor(np.array([1,2.],[3,4])) #从数组中创建张量



#创建全0或全1张量
tf.zeros([]),tf.ones([])#创建全0，全1的标量
tf.zeros([1]),tf.ones(1)#创建全0，全1的向量
tf.zeros([2,2])#创建全0矩阵，指定shape为2行2列
tf.ones([3,2])#创建全1矩阵，指定shape为3行2列

a = tf.ones([2,3])#创建一个矩阵
tf.zeros_like(a)#创建一个与a形状相同，但是全0的新矩阵
tf.ones_like(a)#创建一个与a形状相同，但是全1的新矩阵


#创建自定义数值张量
tf.fill([],-1)#创建-1的标量
tf.fill([1],-1)#创建-1的向量
tf.fill([2,2],99)#创建2行2列，元素全为99的矩阵



#创建已知分布的张量
tf.random.normal([2,2])#创建正态分布的张量
tf.random.normal([2,2],mean = 1,steddev = 2)#创建均值为1，标准差为2的正态分布
tf.random.uniform([2,2])#创建采样自[0,1)均匀分布的矩阵
tf.random.uniform([2,2],maxval = 10)#创建采样自[0,10)均匀分布的矩阵
tf.random.uniform([2,2],maxval = 100,dtype = tf.int32)#创建采样自[0,100)均匀分布的整型矩阵


#创建序列
tf.range(10)#创建0~10,步长为1的整型序列，不包含10
tf.range(10,delta = 2)#创建0~10，步长为2的整型序列
tf.range(1,10,delta = 2)#tf.range(start,limit,delta=1)




out = tf.random.uniform([4,10])#随机模拟网络输出
y = tf.constant([2,3,2,0])#随机构造样本真实标签
y = tf.one_hot(y,depth = 10)#one-hot编码
loss = tf.keras.losses.mse(y,out)#计算每个样本的MSE
loss = tf.reduce_mean(loss)#平均MSE，loss应是标量
print(loss)



#z = wx,模拟获得激活函数的输入z
z = tf.random.normal([4,2])
b = tf.zeros([2])#创建偏置向量
z = z + b #累加偏置向量



fc = layers.Dense(3)#创建一层Wx+b,输出节点为3
#通过build函数创建w,b张量，输入节点为4
fc.build(input_shape = (2,4))
fc.bias#查看偏置向量


#矩阵
x = tf.random.normal([2,4])#2个样本，特征长度为4的张量
w = tf.ones([4,3])#定义W张量
b = tf.zeros([3])#定义b张量
o = x @ w + b #x@w+b运算


fc = layers.Dense(3)#定义全连接层的输出节点为3
fc.build(input_shape = (2,4))#定义全连接层的输入节点为4
fc.kernel #查看权值矩阵W



#三维张量
#自动加载IMDB电影评价数据集
(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words = 10000)
#将句子填充，截断为等长80个单词的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen = 80)
x_train.shape
#创建词向量Embedding层类
embedding = layers.Embedding(10000,100)
#将数字编码的单词转换为词向量
out = embedding(x_train)
out.shape



#四维张量
#创建32*32的彩色图片输入，个数为4
x = tf.random.normal([4,32,32,3])
#创建卷积神经网络
layers = layers.Conv2D(16,kernel_size = 3)
out = layer(x)#前向计算
out.shape#输出大小




#索引
x = tf.random.normal([4,32,32,3])#创建4D张量


#切片
x = tf.range(9)#创建0~9向量
x[8:0:-1]#从8取到0，逆序，不包含0
x[::-1]#逆序全部元素
x[::-2]#逆序间隔采样
x = tf.random.normal([4,32,32,3])
x[0,::-2,::-2]#行，列逆序间隔采样
x[:,:,:,1]#取G通道数据
x[0:2,...,1:]#读取第1~2张图片的G/B通道数据，高宽维度全部采集
x[2:,...]#高、宽、通道维度全部采集，等价于x[2:]
x[...,:2]#所有样本，所有高、宽的前2个通道
x = tf.range(96)#生成向量
x = tf.reshape(x,[2,4,4,3])#改变x的视图，获得4D张量，存储并未改变
x,ndim,x.shape#获取张量的维度数和形状列表
tf.reshape(x,[2,-1])
tf.reshape(x,[2,4,12])
tf.reshape(x,[2,-1,3])


#增删维度
#产生矩阵
x = tf.random.uniform([28,28],maxval = 10,dtype = tf.int32)
x = tf.expand_dims(x,axis = 2)#axis = 2表示宽维度后面的一个维度
x = tf.expand_dims(x,axis = 0)#高维度之前插入新维度
x = tf.squeeze(x,axis = 0)#删除图片数量维度
x = tf.squeeze(x,axis = 2)#删除图片通道数维度
x = tf.random.uniform([1,28,28,1],maxval = 10,dtype = int32)
tf.squeeze(x)#删除所有长度为1的维度



#交换维度
x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm = [0,3,1,2])#交换维度
x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm = [0,2,1,3])#交换维度

b = tf.constant([1,2])#创建向量b
b = tf.expand_dims(b,axis = 0)#插入新维度，变成矩阵
b = tf.tile(b,multiples = [2,1])#样本维度上复制一份
x = tf.range(4)
x = tf.reshape(x,[2,2])#创建2行2列矩阵
x = tf.tile(x,multiples = [1,2])#列维度复制一份
x = tf.tile(x,multiples = [2,1])#行维度复制一份


x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.random.normal([3])
y = x@w + b #不同shape的张量直接相加
y = x@w + tf.broadcast_to(b,[2,3])#手动扩展，并相加


A = tf.random.normal([32,1])#创建矩阵
tf.broadcast_to(A,[2,32,32,3])#扩展为4D张量

A = tf.random.normal([32,2])
tf.broadcast_to(A,[2,32,32,4])#不符合Broadcasting条件

a = tf.random.normal([2,32,32,1])
b = tf.random.normal([32,32])
a+b,a-b,a*b,a/b #测试加减乘除的Brocastcasting机制


#加减乘除运算
a = tf.range(5)
b = tf.constant(2)
a//b #整除运算
a%b #余除运算
#乘方运算
x = tf.range(4)
tf.pow(x,3)#乘方运算
x ** 2#乘方运算符

x = tf.constant([1.,4.,9.])
x ** (0.5)#平方根
x = tf.square(x)#平方
tf.sqrt(x)#平方根



#指数和对数运算
x = tf.constant([1.,2.,3.])
2 ** x#指数运算
tf.exp(1.)#自然指数运算
x = tf.exp(3.)
tf.math.log(x)#对数运算


x = tf.constant([1.,2.])
x = 10 ** x 
tf.math.log(x)/tf.math.log(10.)#换底公式


a = tf.random.normal([4,3,28,32])
b = tf.random.normal([4,3,32,2])
a@b #批量形式的矩阵相乘

a = tf.random.normal([4,28,32])
b = tf.random.normal([32,16])
tf.matmul(a,b)#先自动扩展，再矩阵相乘

