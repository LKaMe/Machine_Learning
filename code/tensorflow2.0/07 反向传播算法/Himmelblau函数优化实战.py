import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def himmelblau(x):
    #himmelblau函数实现，传入参数x为2个元素的List
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6,6,0.1)#可视化的x坐标范围为-6~6
y = np.arange(-6,6,0.1)#可视化的y坐标范围为-6！6
print('x,y range:',x.shape,y.shape)
#生成x-y平面采样网格点，方便可视化
X,Y = np.meshgrid(x,y)
print('X,Y maps:',X.shape,Y.shape)
Z = himmelblau([X,Y])#计算网格点上的函数值
#绘制himmelblau函数曲面
fig = plt.figure('himmelblau')
ax = fig.gca(projection = '3d')#设置3D坐标轴
ax.plot_surface(X,Y,Z)#3D曲面图
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()