import tensorflow as tf
import panas as pd

#我们采用Auto MPG数据集，它记录了各种汽车效能指标与气缸数、重量、马力等其他因子的真实数据
#Auto MPG数据集一共记录了398项数据，我们从UCI服务器下载并读取数据集到DataFrame对象中

#在线下载汽车效能数据集
dataset_path = keras.utils.get_file("auto-mpg.data",
"http://archive.ices.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#利用pandas读取数据集，字段有效能(公里数每加仑)，气缸数，排量，马力，重量
#加速度，型号年份，产地
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
'Acceleration','Model Year','Origion']
raw_dataset = pd.read_csv(dataset_path,names = column_names,
na_values = "?",comment = '\t',
sep = " ",skipinitialspace = True)
dataset = raw_dataset.copy()
#查看部分数据
dataset.head()
dataset.isna().sum()#统计空白数据
dataset = dataset.dropna()#删除空白数据项
dataset.isna().sum()#再次统计空白数据

#由于Origion字段为类别类型数据，我们将其移除，并转换为新的3个字段：USA、Europe和Japan
#处理类别型数据，其中origion列代表了类别1，2，3，分布代表产地：美国、欧洲、日本
#先弹出(删除并返回)origion这一列
origion = dataset.pop('Origion')
#根据origion列来写入新的3个列
dataset('USA') = (origion == 1) * 1.0
dataset['Europe'] = (origion == 2) * 2.0
dataset['Japan'] = (origion == 3) * 1.0 
dataset.tail()#查看新表格的后几项

#接着按8：2的比例切分为数据集为训练集和测试集：
#切分为训练集和测试集
train_dataset = dataset.sample(frac = 0.8,random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

#将MPG字段移出为标签数据：
#移动MPG油耗效能这一列为真实标签Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#统计训练集的各个字段数值的均值和标准差，并完成数据的标准化，通过norm()函数实现
#查看训练集的输入x的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG")#仅保留输入x
train_stats = train_stats.transpose()#转置
#标准化数据
def norm(x):#减去每个字段的均值，并除以标准差
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)#标准化训练集
normed_test_data = norm(test_dataset)#标准化测试集

#打印出训练集和测试集的大小
print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape,test_labels.shape)

#利用切分的训练集数据构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,
train_labels.values))#构建dataset对象
train_db = train_db.shuffle(100).batch(32)#随机打散，批量化




#创建网络
class Network(keras.Model):
    #回归网络模型
    def __init__(self):
        super(Network,self).__init__()
        #创建3个全连接层
        self.fcl = layers.Dense(64,activation = 'relu')
        self.fc2 = layers.Dense(64,activation = 'relu')
        self.fc3 = layers.Dense(1)

    def call(self,inputs,training = None,mask = None):
        #依次通过3个全连接层
        x = self.fcl(inputs)
        x = self.fc2(x)
        x = sel.fc3(x)

        return x



#训练与测试
model = Network()#创建网络类实例
#通过build函数完成内部张量的创建，其中4为任意设置的batch数量，9为输入特征长度
model.build(input_shape = (4,9))
model.summary()#打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001)#创建优化器，指定学习率

for epoch in range(200):#200个Epoch
    for step,(x,y) in enumerate(train_db):#遍历一次训练集
        #梯度记录器，训练时需要使用它
        with tf.gradientTape() as tape:
            out = model(x)#通过网络获得输出
            loss = tf.reduce_mean(losses.MSE(y,out))#计算MSE
            mae_loss = tf.reduce_mean(losses.MAE(y,out))#计算MAE

        if step % 10 == 0:#间隔性地打印训练误差
            print(epoch,step,float(loss))
        #计算梯度，并更新
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
