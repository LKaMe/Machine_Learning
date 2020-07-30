import tensorflow as tf 
def gradient_penalty(discriminator,batch_x,fake_image):
     #梯度惩罚项计算函数
    batchsz = batch_x.shape[0]

     #每个样本均随机采样t,用于插值
    t = tf.random.uniform([batchsz,1,1,1])
     #自动扩展为x的形状，[b,1,1,1] => [b,h,w,c]
    t = tf.broadcast_to(t,batch_x.shape)
     #在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
     #在梯度环境中计算D对插值样本的梯度

    with tf.GradientTape() as tape:
        tape.watch([interplate])#加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits,interplate)

    #计算每个样本的梯度的范数[b,h,w,c] => [b,-1]
    grads = tf.reshape(grads,[grads.shape[0],-1])
    gp = tf.norm(grads,axis = 1) #[b]
    #计算梯度惩罚项
    gp = tf.reduce_mean((gp-1.) ** 2)

    return gp 

def d_loss_fn(generator,discriminator,batch_z,batch_x,is_training):
    #计算D的损失函数
    fake_image = generator(batch_z,is_training)#假样本
    d_fake_logits = discriminator(fake_image,is_training)#假样本的输出
    d_real_logits = discriminator(batch_x,is_training)#真样本的输出
    #计算梯度惩罚项
    gp = gradient_penalty(discriminator,batch_x,fake_image)
    #WGAN-GP D损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    #最小化假样本的输出和梯度惩罚项
    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10. * gp 

    return loss,gp 



