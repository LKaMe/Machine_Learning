#VAE模型
class VAE(keras.Model):
    #变分自编码器
    def __init__(self):
        super(VAE,self).__init__()
        #Encoder网络
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)#均值输出
        self.fc3 = layers.Dense(z_dim)#方差输出

        #Decoder网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    #encoder的输入先通过共享层fc1，然后分别通过fc2与fc3网络，获得隐向量分布的均值向量与方差的log向量值
    def encoder(self,x):
        #获得编码器的均值和方差
        h = tf.nn.relu(self.fc1(x))
        #均值向量
        mu = self.fc2(h)
        #方差的log向量
        log_var = self.fc3(h)

        return mu,log_var 
    #Decoder接受采样后的隐向量z,并解码为图片输出
    def decoder(self,z):
        #根据隐藏变量z生成图片数据
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        #返回图片数据，784向量
        return out 
    def call(self,inputs,training = None):
        #前向计算
        编码器[b,784] => [b,z_dim],[b,z_dim]
        mu,log_var = self.encoder(inputs)
        #采样reparameterization trick
        z = self.reparameterize(mu,log_var)
        #通过解码器生成
        x_hat = self.decoder(z)
        #返回生成样本，及其均值与方差
        return x_hat,mu,log_var 
    
    #Reparameterization技巧
    def reparameterize(self,mu,log_var):
        #reparameterize技巧，从正态分布采样epsion
        eps = tf.random.normal(log_var.shape)
        #计算标准差
        std = tf.exp(log_var) ** 0.5
        #reparameterize技巧
        z = mu + std * eps 
        return z 
    
    #网络训练
    #创建网络对象
    model = VAE()
    model.build(input_shape = (4,784))
    #优化器
    optimizer = optimizers.Adam(lr)

    for epoch in range(100):#训练100个Epoch
        for step,x in enumerate(train_db):#遍历训练集
            #打平，[b,28,28] => [b,784]
            x = tf.reshape(x,[-1,784])
            #构建梯度记录器
            with tf.GradientTape() as tape:
                #前向计算
                x+rec_logits,mu,log_var = model(x)
                #重建损失值计算
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = x,logits = x_rec_logits)
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
                #计算KL散度N(mu,var) VS N(0,1)
                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]
                #合并误差项
                loss = rec_loss + 1. * kl_div 
            #自动求导
            grads = tape.gradient(loss,model.trainable_variables)
            #自动更新
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            if step % 100 == 0:
                #打印训练误差
                print(epoch,step,'kl div:',float(kl_div),'rec loss:',float(rec_loss))

    #图片生成
    #测试生成效果，从正态分布随机采样z
    z = tf.random.normal((batchsz,z_dim))            
    logits = model.decoder(z)#仅通过解码器生成图片
    x_hat = tf.sigmoid(logits)#转换为像素范围
    x_hat = tf.reshape(x_hat,[-1,28,28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat,'vae_images/epoch_%d_sampled.png'%epoch)#保存生成图片

    #重建图片，从测试集采样图片
    x = next(iter(test_db))
    logits,_,_=model(tf.reshape(x,[-1,784]))#打平并送入自编码器
    x_hat = tf.sigmoid(logits)#将输出转换为像素值
    #恢复为28*28，[b,784] => [b,28,28]
    x_hat = tf.reshape(x_hat,[-1,28,28])
    #输入的前50张+重建的前50张图片合并，[b,28,28] => [2b,28,28]
    x_concat = tf.concat([x[:50],x_hat[:50]],axis = 0)
    x_concat = x_concat.numpy() * 255. #恢复为0~255范围
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat,'vae_images/epoch_%d_rec.png'%epoch)#保存重建图片
    
