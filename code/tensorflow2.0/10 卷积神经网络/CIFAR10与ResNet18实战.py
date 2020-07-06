import tensorflow as tf 
class BasicBlock(layers.Layer):
    #残差模块
    def __init__(self,filter_num,stride = 1):
        super(BasicBlock,self).__init__()
        #第一个卷积单元
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides = stride,padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        #第二个卷积单元
        self.conv2 = layers.Conv2D(filter_num,(3,3),strides = 1,padding = 'same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:#通过1*1卷积完成shape匹配
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:#shape匹配，直接连接
            self.downsample = lambda x:x 

        def call(self,inputs,training = None):
            #前向计算函数
            #[b,h,w,c]通过第一个卷积单元
            out = self.conv1(inputs)
            out = self.bn1(out)
            out = self.relu(out)
            #通过第二个卷积单元
            out = self.conv2(out)
            out = self.bn2(out)
            #通过identity模块
            identity = self.downsample(inputs)
            #2条路径输出直接相加
            output = layers.add([out,identity])
            output = tf.nn.relu(output)#激活函数

            return output 

        def build_resblock(self,filter_num,blocks,stride = 1):
            #辅助函数，堆叠filter_num个BasicBlock 
            res_blocks = Sequential()
            #只有第一个BasicBlock的步长可能不为1，实现下采样
            res_blocks.add(BasicBlock(filter_num,stride))

            for _ in range(1,blocks):#其他BasicBlock步长都为1 
                res_blocks.add(BasicBlock(filter_num,stride = 1))

            return res_blocks 

    
    class ResNet(keras.Model):
        #通用的ResNet实现类
        def __init__(self,layer_dims,num_classes=10):#[2,2,2,2]
            super(ResNet,self).__init__()
            #根网络，预处理
            self.stem = Sequential([layers.Conv2D(64,(3,3),strides = (1,1)),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPool2D(pool_size = (2,2),strides = (1,1),padding = 'same')
                                ])
            #堆叠4个Block,每个block包含了多个BasicBlock,设置步长不一样
            self.layer1 = self.build_resblock(64,layer_dims[0])
            self.layer2 = self.build_resblock(128,layer_dims[1],stride=2)
            self.layer3 = self.build_resblock(256,layer_dims[2],stride=2)
            self.layer4 = self.build_resblock(512,layer_dims[3],stride=2)

            #通过Pooling层将高宽降低为1*1
            self.avgpool = layers.G;lobalAveragePooling2D()
            #最后连接一个全连接层分类
            self.fc = layers.Dense(num_classes)

        def call(self,inputs,training = None):
            #前向计算函数，通过根网络
            x = self.stem(inputs)
            #一次通过4个模块
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            #通过池化层
            x = self.avgpool(x)
            #通过全连接层
            x = self.fc(x)

            return x 
        
        def resnet18():
            #通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
            return ResNet([2,2,2,2])

        def resnet34():
            #通过调整模块内部BasicBlock的数量和配置实现不同的ResNet 
            return ResNet([3,4,6,3])

        #CIFAR10数据集加载工作
        (x,y),(x_test,y_test) = datasets.cifar10.load_data()#加载数据集
        y = tf.squeeze(y,axis = 1)#删除不必要的维度
        y_test = tf.squeeze(y_test,axis = 1)#删除不必要的维度
        print(x.shape,y.shape,x_test.shape,y_test.shape)

        train_db = tf.data.Dataset.from_tensor_slices((x,y))#构建训练集
        #随机打散，预处理，批量化
        train_db = train_db.shuffle(1000).map(preprocess).batch(512)

        test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))#构建测试集
        #随机打散，预处理，批量化
        test_db = test_db.map(preprocess).batch(512)
        #采样一个样本
        sample = next(iter(train_db))
        print('sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))

        def preprocess(x,y):
            #将数据映射到-1~1
            x = 2 * tf.cast(x,dtype = tf.float32) / 255. - 1
            y = tf.cast(y,dtype = tf.int32)#类型转换
            return x,y 

            for epoch in range(50):#训练epoch 
                for step,(x,y) in enumerate(train_db):
                    with tf.GradientTape() as tape :
                        #[b,32,32,3] => [b,10],前向传播
                        logits = model(x)
                        #[b] => [b,10],one-ht编码
                        y_onehot = tf.one_hot(y,depth = 10)
                        #计算交叉熵
                        loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits = True)
                        loss = tf.reduce_mean(loss)
                    #计算梯度信息
                    grads = tape.gradient(loss,model.trainable_variables)
                    #更新网络参数
                    optimizer.apply_gradients(zip(grads,model.trainable_varoables))
                    

