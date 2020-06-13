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

