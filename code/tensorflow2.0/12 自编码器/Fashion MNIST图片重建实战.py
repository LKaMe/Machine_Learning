import tensorflow as tf 

#加载Fashion MNIST图片数据集
(x_train,y_train),(x_test,y_test) = keras.datasets.fashion_mnist.load_data()
#归一化
x_train,x_test = x_train.astype(np.float32) / 255.,x_test.astype(np.float32) / 255.
#只需要通过图片数据即可构建数据集对象，不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
#构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

#创建Encoders网络，实现在自编码器类的初始化函数中
self.encoder = Sequential([
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(h_dim)
])

#创建Decoders网络
self.decoder = Sequential([
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(784)
])

#自编码器
class AE(keras.Model):
    #自编码器模型类，包含了Encoder和Decoder2个子网络
    def __init__(self):
        super(AE,self).__init__()
        #创建Encoder网络
        self.encoder = Sequential([
            layers.Dense(256,activation = tf.nn.relu),
            layers.Dense(128,activation = tf.nn.relu),
            layers.Dense(h_dim)
        ])
        #创建Decoder网络
        self.decoder = Sequential([
            layers.Dense(128,activation = tf.nn.relu),
            layers.Dense(256,activation = tf.nn.relu),
            layers.Dense(784)
        ])
        def call(self,inputs,training = None):
            #前向传播函数
            #编码获得隐藏向量h,[b,784] => [b,20]
            h = self.encoder(inputs)
            #解码获得重建图片，[b,20] => [b,784]
            x_hat = self.decoder(h)

            return x_hat 

#创建网络对象
model = AE()
#指定输入大小
model.build(input_shape = (4,784))
#打印网络信息
model.summary()
#创建优化器，并设置学习率
optimizer = optimizers.Adam(lr = lr)

for epoch in range(100):#训练100个Epoch
    for step,x in enumerate(train_db):#遍历训练集
        #打平，[b,28,28] => [b,784]
        x = tf.reshape(x,[-1,784])
        #构建梯度记录器
        with tf.GradientTape() as tape:
            #前向计算获得重建的图片
            x_rec_logits = model(x)
            #计算重建图片与输入之间的损失函数
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = x,logits = x_rec_logits)
            #计算均值
            rec_loss = tf.reduce_mean(rec_loss)
        #自动求导，包含了2个子网络的梯度
        grads = tape.gradient(rec_loss,model.trainable_variables)
        #自动更新，同时更新2个子网络
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step % 100 == 0:
            #间隔性打印训练误差
            print(epoch,step,float(rec_loss))

#图片重建
#重建图片，从测试集采样一批图片
x = next(iter(test_db))
logits = model(tf.reshape(x,[-1,784]))#打平并送入自编码器
x_hat = tf.sigmoid(logits)#将输出转换为像素值，使用sigmoid函数
#恢复为28*28，[b,784] => [b,28,28]

#输入的前50张+重建的前50张图片合并,[b,28,28] => [2b,28,28]
x_concat = tf.concat([x[:50],x_hat[:50]],axis = 0)
x_concat = x_concat.numpy() * 255. #恢复为0~255范围
x_concat = x_caoncat.astype(np.uint8)#转换为整型
save_images(x_concat,'ae_images/rec_epoch_%d.png'%epoch)#保存图片

#使用PIL图片库完成图片阵列逻辑
def save_images(imgs,name):
    #创建280*280大小图片阵列
    new_im = Image.new('L',(280,280))
    index = 0
    for i in range(0,280,28):#10行图片阵列
        for j in range(0,280,28):#10列图片阵列
            im = imgs[index]
            im = Image.fromarray(im,mode = 'L')
            new_im.paste(im,(i,j))#写入对应位置
            index += 1
    #保存图片阵列
    new_im.save(name)
    
