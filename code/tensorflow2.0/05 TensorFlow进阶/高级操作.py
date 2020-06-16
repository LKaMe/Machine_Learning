import tensorflow as tf


#tf.gather函数
x = tf.random.uniform([4,35,8],maxval = 100,dtype = tf.int32)#成绩册张量
tf.gather(x,[0,1],axis = 0)#在班级维度收集第1~2号班级成绩册
#收集第1，4，9，12，13，27号同学成绩
tf.gather(x,[0,3,8,11,12,26],axis = 1)
tf.gather(x,[2,4],axis = 2)#第3，5科目的成绩

a = tf.range(8)
a = tf.reshape(a,[4,2])#生成张量a
tf.gather(a,[3,1,0,2],axis = 0)#收集第4，2，1，3号元素

students = tf.gather(x,[1,2],axis = 0)#收集第2，3号班级
tf.gather(students,[2,3,5,26],axis = 1)#收集第3，4，6，27号同学
x[1,1]#收集第2个班级的第2个同学

tf.stack([x[1,1],x[2,2],x[3,3]],axis = 0)


#tf.gather_nd函数
#根据多维坐标收集数据
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
#根据多维度坐标收集数据
tf.gather_nd(x,[[1,1,2],[2,2,3],[3,3,4]])


#tf.boolean_mask
#根据掩码方式采样班级，给出掩码和维度索引
tf.boolean_mask(x,mask = [True,False,False,True],axis = 0)
#根据掩码方式采样科目
tf.boolean_mask(x,mask = [True,False,False,True,True,False,False,True],axis = 2)
x = tf.random.uniform([2,3,8],maxval = 100,dtype = tf.int32)
tf.gather_nd(x,[[0,0],[0,1],[1,1],[1,2]])#多维坐标采集
#多维掩码采样
tf.boolean_mask(x,[[True,True,False],[False,True,True]])




#tf.where
a = tf.ones([3,3])#构造a为全1矩阵
b = tf.zeros([3,3])#构造b为全0矩阵
#构造采样条件
cond = tf.constant([[True,False,False],[False,True,False],[True,True,False]])
tf.where(cond,a,b)#根据条件从a,b中采样
cond#构造的cond张量
tf.where(cond)#获取cond中为True的元素索引
x = tf.random.normal([3,3])#构造a
mask = x > 0#比较操作，等同于tf.math.greater()
mask
indices = tf.where(mask)#提取所有大于0的元素索引
tf.gather_nd(x,indices)#提取正数的元素值
tf.boolean_mask(x,mask)#通过掩码提取正数的元素值



#构造需要刷新数据的位置参数，即为4，3，1和7号位置
indices = tf.constant([[4],[3],[1],[7]])
#构造需要写入的数据，4号位写入4，4，3号位写入3.3，以此类推
updates = tf.constant([4,4,3.3,1.1,7.7])
#在长度为8的全0向量上根据indices写入updates数据
tf.scatter_nd(indices.updates,[8])


#构造写入位置，即2个位置
indices = tf.constant([1],[3])
updates = tf.constant([#构造写入数据，即2个矩阵
    [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],
    [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
])
#在shape为[4,4,4]白板上根据indices写入updates
tf.scatter_nd(indices,updates,[4,4,4])



#meshgrid
points = []#保存所有点的坐标列表
for x in range(-8,8,100):#循环生成x坐标，100个采样点
    for y in range(-8,8,100):#循环生成y坐标，100个采样点
        z = sinc(x,y)#计算每个点(x,y)处的sinc函数值
        points.append([x,y,z])#保存采样点


x = tf.linspace(-8,8,100)#设置x轴的采样点
y = tf.linspace(-8,8,100)#设置y轴的采样点
x,y = tf.meshgrid(x,y)#生成网格点，并内部拆分后返回
x.shape,y.shape#打印拆分后的所有点的x,y坐标张量shape
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z#sinc函数实现




import matplotlib 
from matplotlib import pyplot as plt 
#导入3D坐标轴支持
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)#设置3D坐标轴
#根据网格点绘制sinc函数3D曲面
ax.contour3D(x.numpy(),y.numpy(),z.numpy(),50)
plt.show()