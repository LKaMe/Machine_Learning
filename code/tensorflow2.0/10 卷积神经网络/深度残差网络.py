#ResBlock实现
class __init__(self,filter_num,strides = 1):
    super(BasicBlock,self).__init__()
    #f(x)包含了2个普通卷积层，创建卷积层1
    self.conv1 = layers.Conv2D(filter_num,(3,3),strides = stride,padding = 'same')
    self.bn1 = layers.BatchNormalization()
    self.relu = layers.Activation('relu')
    #创建卷积层2
    self.conv2 = layers.Conv2D(filter_num,(3,3),strides = 1,padding = 'same')
    self.bn2 = layers.BatchNormalization()

    if stride != 1:#插入identity层
        self.downsample = Sequential()
        self.downsample.add(layers.Conv2D(filter_num,(1,1),strides = stride))
    else:#否则，直接连接
        self.downsample = lambda x:x 

    def call(self,inputs,training = None):
        #前向传播函数
        out = self.conv1(inputs)#通过第一个卷积层
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)#通过第二个卷积层
        out = self.bn2(out)
        #输入通过identity()转换
        identity = self.downsample(inputs)
        #f(x)+x运算
        output = layers.add([out,identity])
        #再通过激活函数并返回
        output = tf.nn.relu(output)
        return output 
        