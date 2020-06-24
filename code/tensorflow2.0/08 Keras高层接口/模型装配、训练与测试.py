import tensorflow
from tensorflow.keras import optimizers,losses
#模型装配
#以Sequential容器封装的网络为例，首先创建5层的全连接网络，用于MNIST手写数字图片识别
#创建5层的全连接网络
# network = Sequential([layers.Dense(256,activation='relu'),
# layers.Dense(128,activation = 'relu'),
# layers.Dense(64,activation = 'relu'),
# layers.Dense(32,activation = 'relu'),
# layers.Dense(10)])
# network.build(input_shape = (4,28 * 28))
# network.summary()

#模型装配
#采用Adam优化器，学习率为0.01;采用交叉熵损失函数，包含Softmax
network.compile(optimizer = optimizers.Adam(lr = 0.01),
loss = losses.CategoricalCrossentropy(from_logits=True),
metrics = ['accuracy']#设置测量指标为准确率
)

#模型训练
#指定训练集为train_db,验证集为val_db,训练5个epochs,每2个epoch验证一次
#返回训练轨迹信息保存在history对象中
history = network.fit(train_db,epochs = 5,validation_data = val_db,validation_freq = 2)
print(history.history)#打印训练记录


#模型测试
#加载一个batch的测试数据
x,y = next(iter(db_test))
print('predict x:',x,shape)#打印当前batch的形状
out = network.predict(x)#模型预测，预测结果保存在out中
print(out)
network.evaluate(db_test)#模型测试，测试在db_test上的性能表现

