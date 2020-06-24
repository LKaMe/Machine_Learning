import tensorflow as tf 
from tensorflow import keras 
#张量方式
#保存模型参数到文件上
network.save_weights('weights.ckpt')
print('saved weights.')
del network #删除网络对象
#重新创建相同的网络结构
network = Sequential([layers.Dense(256,activation = 'relu'),
    layers.Dense(128,activation = 'relu'),
    layers.Dense(64,activation = 'relu'),
    layers.Dense(32,activation = 'relu'),
    layers.Dense(10)])
network.compile(optimizer = optimizer.Adam(lr = 0.1),
loss = tf.losses.CategoricalCrossentropy(from_logits = True),
metrics = ['accuracy'])

#从参数文件中读取数据并写入当前网络
network.load_weights('weights.ckpt')
print('loaded weights!')

#网络方式
#保存模型结构与模型参数到文件
network.save('model.h5')
print('saved total model.')
del network #删除网络对象
#从文件恢复网络结构与网络参数
network = keras.models.load_model('model.h5')
#SavedModel方式
#保存模型结构与模型参数到文件
tf.saved_model.save(network,'model-saved model')
print('saving saved model')
del network  #删除网络对象
print('load saved model from file.')
#从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel')
#准确率计量器
acc_meter = metrics.CategoricalAccuracy()
for x,y in ds_val:#遍历测试集
    pred = network(x)#前向计算
    acc_meter.update_state(y_true = y,y_pred = pred)#更新准确率统计
#统计准确率
print("Test Accuracy:%f" % acc_meter.result())
