import tensorflow as tf 
# 数据集路径，从 https://pan.baidu.com/s/1eSifHcA 提取码：g5qa 下载解压
img_path = glob.glob(r'xxx')
#构建数据集对象，返回数据集Dataset类和图片大小
dataset,img_shape,_=make_anime_dataset(img_path,batch_size,resize = 64)

#生成器
class Generator(keras.Model):
    #生成器网络类
    def __init__(self):
        super(Generator,self).__init__()
        filter = 64
        #转置卷积层1，输出cnannel为filter * 8,核大小4，步长1，不使用padding,不使用偏置
        self.conv1 = layers.Conv2DTranspose(filter * 8,4,1,'valid',use_bias = False)
        self.bn1 = layers.BatchNormalization()
        #转置卷积层2
        self.conv2 = layers.Conv2DTranspose(filter*4,4,2,'same',use_bias = False)
        self.bn2 = layers.BatchNormalization()
        #转置卷积层3
        self.conv3 = layers.Conv2DTranspose(filter*2,4,2,'same',use_bias = False)
        self.bn3 = layers.BatchNormalization()
        #转置卷积层4
        self.conv4 = layers.Conv2DTranspose(filter*1,4,2,'same',use_bias = False)
        self.bn4 = layers.BatchNormalization()
        #转置卷积层5
        self.conv5 = layers.Conv2DTranspose(3,4,2,'same',use_bias = False)

    def call(self,inputs,training = None):
        x = inputs #[z,100]
        #Reshape乘4D张量，方便后续转置卷积运算:(b,1,1,100)
        x = tf.reshape(x,(x.shape[0],1,1,x.shape[1]))
        x = tf.nn.relu(x)#激活函数
        #转置卷积-BN-激活函数:(b,4,4,512)
        x = tf.nn.relu(self.bn2(self.conv2(x),training = training))
        #转置卷积-BN-激活函数:(b,16,16,128)
        x = tf.nn.relu(self.bn3(self.conv3(x),training=training))
        #转置卷积-BN-激活函数:(b,32,32,64)
        x = tf.nn.relu(self.bn4(self.conv4(x),training=training))
        #转置卷积-激活函数:(b,64,64,3)
        x = self.conv5(x)
        x = tf.tanh(x)#输出x范围-1~1，与预处理一致

        return x 

#判别器
class Discriminator(keras.Model):
    #判别器类
    def __init__(self):
        super(Discriminator,self).__init__()
        filter = 64
        #卷积层1
        self.conv1 = layers.Conv2D(filter,4,2,'valid',use_bias = False)
        self.bn1 = layers.BatchNormalization()
        #卷积层2
        self.conv2 = layers.Conv2D(filter*2,4,2,'valid',use_bias = False)
        self.bn2 = layers.BatchNormalization()
        #卷积层3
        self.conv3 = layers.Conv2D(filter*4,4,2,'valid',use_bias = False)
        self.bn3 = layers.BatchNormalization()
        #卷积层4
        self.conv4 = layers.Conv2D(filter*8,3,1,'valid',use_bias = False)
        self.bn4 = layers.BatchNormalization()
        #卷积层5
        self.conv5 = layers.Conv2D(filter*16,3,1,'valid',use_bias = False)
        self.bn5 = layers.BatchNormalization()
        #全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        #特征打平层
        self.flatten = layers.Flatten()
        #2分类全连接层
        self.fc = layers.Dense(1)
    #判别器D的前向计算过程实现如下
    def call(self,inputs,training = None):
        #卷积-BN-激活函数：(4,31,31,64)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))

        #卷积-BN-激活函数:(4,14,14,128)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x),training = training))
        #卷积-BN-激活函数:(4,6,6,256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x),training = training))
        #卷积-BN-激活函数:(4,4,4,512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x),training = training))
        #卷积-BN-激活函数:(4,2,2,1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x),training = training))
        #卷积-BN-激活函数:(4,1024)
        x = self.pool(x)
        #打平
        x = self.flatten(x)
        #输出，[b,1024] => [b,1]
        logits = self.fc(x)

        return logits
    
    #训练与可视化
    def d_loss_fn(generator,discriminator,batch_z,batch_x,is_training):
        #计算判别器的误差函数
        #采样生成图片
        fake_image = generator(batch_z,is_training)
        #判定生成图片
        d_fake_logits = discriminator(fake_image,is_training)
        #判定真实图片
        d_real_logits = discriminator(batch_x,is_training)
        #真实图片与1之间的误差
        d_loss_real = celoss_ones(d_real_logits)
        #生成图片与0之间的误差
        d_loss_fake = celoss_zeros(d_fake_logits)
        #合并误差
        loss = d_loss_fake + d_loss_real 

        return loss 
    
    def celoss_ones(logits):
        #计算属于与标签为1的交叉熵
        y = tf.ones_like(logits)
        loss = keras.losses.binary_crossentropy(y,logits,from_logits = True)
        return tf.reduce_mean(loss)

    def celoss_zeros(logits):
        #计算属于与便签为0的交叉熵
        y = tf.zeros_like(logits)
        loss = keras.losses.binary_crossentropy(y,logits,from_logits = True)
        return tf.reduce_mean(loss)

    def g_loss_fn(generator,discriminator,batch_z,is_training):
        #采样生成图片
        fake_image = generator(batch_z,is_training)
        #在训练生成网络时，需要迫使生成图片判定为真
        d_fake_logits = discriminator(fake_image,is_training)
        #计算生成图片与1之间的误差
        loss = celoss_ones(d_fake_logits)

        return loss 

    #网络训练
    generator = Generator()#创建生成器
    generator.build(input_shape = (4,z_dim))
    Discriminator = Discriminator()#创建判别器
    Discriminator.build(input_shape = (4,64,64,3))
    #分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate = learning_rate,beta_1 = 0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate = learning_rate,beta_1 = 0.5)

    #主要训练部分代码实现如下
    for epoch in range(epochs):#训练epochs次
        #1.训练判别器
        for _ in range(5):
            #采样隐藏向量
            batch_z = tf.random.normal([batch_size,z_dim])
            batch_x = next(db_iter) #采样真实图片
            #判别器前向计算
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator,discriminator,batch_z,batch_x,is_training)
            grads = tape.gradient(d_loss,discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))
            #2.训练生成器
            #采样隐藏向量
            batch_z = tf.random.normal([batch_size,z_dim])
            batch_x = next(db_iter)#采样真实图片
            #生成器前向计算
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator,discriminator,batch_z,is_training)
            grads = tape.gradient(g_loss,generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads,generator.trainable.trainable_variables))
            
