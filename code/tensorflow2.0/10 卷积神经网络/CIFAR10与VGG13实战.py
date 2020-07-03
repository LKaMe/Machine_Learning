import tensorflow as tf 
 #在线下载，加载CIFAR10数据集
 (x,y),(x_test,y_test) = datasets.cifar10.load_data()
 #删除y的一个维度,[b,1] => [b]
 y = tf.sequeexe(y,axis = 1)
 y_test = tf.squeeze(y_test,axis = 1)
 #打印训练集和测试集的形状
 print(x.shape,y.shape,x_test.shape,y_test.shape)
 #构建训练集对象，预处理，批量化
 test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
 test_db = test_db.map(preprocess).batch(128)
 #从训练集中采样一个Batch,并观察
 sample = next(iter(train_db))
 print('sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))

 conv_layers = [#先创建包含多网络层的列表
    #Conv-Conv-Pooling单元1
    #64个卷积核，输入输出同大小
    layers.Conv2D(64,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    #高宽减半
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),
    #Conv-conv-pooling单元2，输出通道提升至128，高宽大小减半
    layers.Conv2D(128,Kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.Conv2D(128,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.MaxPooling2D(pool_size = [2,2],strides = 2,padding = 'same'),

    #Conv-conv-pooling单元3，输出通道提升至256，高宽大小减半
    layers.Conv2D(256,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.Conv2D(256,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = "same"),

    #Conv-conv-pooling单元，输出通道提升至512，高宽大小减半
    layers.Conv2D(512,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.Conv2D(512,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),

    #Conv-Conv-Pooling单元5，输出通道提升至512，高宽大小减半
    layers.Conv2D(512,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.Conv2D(512,kernel_size = [3,3],padding = "same",activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same')
 ]

#利用前面创建的层列表构建网络容器
conv_net = Sequential(conv_layers)

#创建3层全连接层子网络
fc_net = Sequential([
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(10,activation = None),
])

#build2个子网络，并打印网络参数信息
conv_net.build(input_shape = [4,32,32,3])
fc_net.build(input_shape = [4,512])
conv_net.summary()
fc_net.summary()

#列表合并，合并2个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables
#对所有参数求梯度
grads = tape.gradient(loss,variables)
#自动更新
optimizer.apply_gradients(zip(grads,variables))


#转置卷积
#创建x矩阵，高宽为5*5
x = tf.range(25) + 1
#Reshape为合法维度的张量
x = tf.reshape(x,[1,5,5,1])
x = tf.cast(x,tf.float32)
#创建固定内容的卷积核矩阵
w = tf.constant([[-1,2,-3.],[4,-5,6],[-7,8,-9]])
#调整为合法维度的张量
w = tf.expand_dims(w,axis = 2)
w = tf.expand_dims(w,axis = 3)
#进行普通卷积运算
out = tf.nn.conv2d(x,w,strides = 2,padding = 'VALID')


#普通卷积的输出作为转置卷积的输入，进行转置卷积运算
xx = tf.nn.conv2d_transpose(out,w,strides = 2,padding = 'VALID',output_shape = [1,5,5,1])

x = tf.random.normal([1,6,6,1])
#6*6的输入经过普通卷积
out = tf.nn.conv2d(x,w,strides = 2,padding = "VALID")
out.shape 
x = tf.random.normal([1,6,6,1])

#恢复出6*6大小
xx = tf.nn.conv2d_transpose(out,w,strides = 2,padding = 'VALID',output_shape = [1,6,6,1])


#转置卷积实现
#创建4*4大小的输入
x = tf.range(16)+1 
x = tf.reshape(x,[1,4,4,1])
x = tf.cast(x,tf.float32)
#创建3*3卷积核
w = tf.constant([[-1,2,-3.],[4,-5,6],[-7,8,-9]])
w = tf.expand_dims(w,axis = 2)
w = tf.expand_dims(w,axis = 3)
#普通卷积运算
out = tf.nn.conv2d(x,w,strides = 1,padding = 'VALID')

#恢复4*4大小的输入
xx = tf.nn.conv2d_tranpose(out,strides = 1,padding = 'VALID',output_shape = [1,4,4,1])
tf.squeeze(xx)

#创建转置卷积类
layer = layers.Conv2DTranspose(1,kernel_size = 3,strides = 1,padding = 'VALID')
xx = layer(out)#通过转置卷积层

