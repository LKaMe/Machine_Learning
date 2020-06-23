#绘制误差函数J的图形
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号,主要是负号的显示

a=np.arange(-1,3,0.1)
N=a.size
Z=np.zeros((N,N))

def error_function(theta):
    '''Error function J definition.'''
    m = 20
    X0 = np.ones((m, 1))
    X1 = np.arange(1, m+1).reshape(m, 1)
    X = np.hstack((X0, X1))  
    # y = np.array([
    # 3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    # 11, 13, 13, 16, 17, 18, 17, 19, 21
    # ]).reshape(m, 1)
    y = np.array([
    4, 6, 7, 8, 2, 4, 7, 9, 11, 10, 12,
    11, 13, 14, 16, 17, 18, 17, 19, 21
    ]).reshape(m, 1)
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

for i in range(0,N):
    for j in range(0,N):
        theta = np.array([[a[i],a[j]]]).reshape(2, 1)
        Z[i][j]=error_function(theta)
    
X, Y = np.meshgrid(a,a)
fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, Z)#没有经过渲染的蓝色图片
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', edgecolor='none')#彩虹色图片
plt.rcParams['font.sans-serif'] = 'SimHei' #用于正常显示中文
plt.title('损失函数')#添加标题
#ax = Axes3D(fig)
# plt.rcParams['font.sans-serif'] = 'SimHei' #用于正常显示中文
# plt.title('损失函数图形')#添加标题
plt.show()