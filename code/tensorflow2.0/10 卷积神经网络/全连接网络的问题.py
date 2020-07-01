import tensorflow as tf 
from tendorflow import keras 
from tensorflow.keras import layers.Sequential,losses,optimizers,datasets 
#创建4层全连接网络
model = keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)
])
#build模型，并打印模型信息
model.build(input_shape=(4,784))
model.summary()

#获取所有GPU设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #设置GPU显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        #异常处理
        print(e)
        