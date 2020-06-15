import tensorflow as tf

#拼接
a = tf.random.normal([4,35,8])#模拟成绩册A
b = tf.random.normal([6,35,8])#模拟成绩册B
tf.concat([a,b],axis = 0)#拼接合并成绩册


a = tf.random.normal([10,35,4])
b = tf.random.normal([10,35,4])
tf.concat([a,b],axis = 2)#在科目维度上拼接


a = tf.random.normal([4,32,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b],axis = 0)#非法拼接，其他维度长度不相同


a = tf.random.normal([4,32,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b],axis = 0)#非法拼接，其他维度长度不相同

#堆叠
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis = 0)#堆叠合并为2个班级，班级维度插入在最前

a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis = -1)#在末尾插入班级维度


a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.concat([a,b],axis = 0)#拼接方式合并，没有2个班级的概念


a = tf.random.normal([35,4])
b = tf.random.normal([35,8])
tf.stack([a,b],axis = -1)#非法堆叠操作，张量shape不相同



#分割
x = tf.random.normal([10,35,8])
#等长切割为10份
result = tf.split(x,num_or_size_splits = 10,axis = 0)
len(result)#返回的列表为10个张量的列表
result[0]#查看第一个班级的成绩册张量

x = tf.random.normal([10,35,8])
#自定义长度的切割，切割为4份，返回4个张量的列表result
result = tf.split(x,num_or_size_splits = [4,2,2,2],axis = 0)
len(result)
result[0]

x = tf.random.normal([10,35,8])
result = tf.unstack(x,axis = 0)#Unstack为长度为1的张量
len(result)#返回10个张量的列表

