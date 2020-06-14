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




