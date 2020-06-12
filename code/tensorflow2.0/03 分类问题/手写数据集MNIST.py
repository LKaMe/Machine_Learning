#利用Tensorflow自动在线下载MNIST数据集，并转换为Numpy数组格式
import os 
import tensorflow as tf #导入TF库
from tensorflow import keras #导入TF子库keras
from tensorflow.keras import layers,optimizers,datasets#导入TF子库等

(x,y),(x_val,y_val) = datasets.mnist.load_data() #加载MNIST数据集
x = 2 * tf.convert_to_tensor(x,dtype = tf.float32)/255.-1 #转换为浮点张量，并缩放到-1~1
y = tf.convert_to_tensor(y,dtype = tf.int32)#转换为整形张量
y = tf.one_hot(y,depth = 10)#one-hot编码
print(x.shape,y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))#构建数据集对象
train_dataset = train_dataset.batch(512)#批量训练
