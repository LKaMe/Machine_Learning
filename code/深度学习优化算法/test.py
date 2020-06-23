# #原文链接：https://blog.csdn.net/qq_41856733/article/details/102677984
# #数据散点图及梯度下降算法拟合出来的直线
# import numpy as np
# import matplotlib.pyplot as plt
# # Size of the points dataset.
# m = 20

# # Points x-coordinate and dummy value (x0, x1).
# X0 = np.ones((m, 1))
# X1 = np.arange(1, m+1).reshape(m, 1)
# X = np.hstack((X0, X1))

# # Points y-coordinate
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)

# '''
# plt.plot(X1,y)
# plt.show()
# plt.scatter(X1,y)
# plt.show()
# '''

# # The Learning Rate alpha.
# alpha = 0.01

# def error_function(theta, X, y):
#     '''Error function J definition.(误差函数J)'''
#     diff = np.dot(X, theta) - y
#     return (1./2*m) * np.dot(np.transpose(diff), diff)

# def gradient_function(theta, X, y):
#     '''Gradient of the function J definition..(误差函数J梯度定义，对theta求导结果)'''
#     diff = np.dot(X, theta) - y
#     return (1./m) * np.dot(np.transpose(X), diff)

# def gradient_descent(X, y, alpha):
#     '''Perform gradient descent.'''
#     theta = np.array([1, 1]).reshape(2, 1)
#     gradient = gradient_function(theta, X, y) #梯度下降最快的方向
#     while not np.all(np.absolute(gradient) <= 1e-5):
#         theta = theta - alpha * gradient
#         gradient = gradient_function(theta, X, y)
#     return theta

# optimal = gradient_descent(X, y, alpha)
# print('optimal:', optimal)
# print('error function:', error_function(optimal, X, y)[0,0])
# #绘制拟合后的图形和散点图
# plt.rcParams['font.sans-serif'] = 'SimHei' #用于正常显示中文
# plt.title('数据散点图及其梯度下降法拟合出的直线') ## 添加标题
# plt.xlabel('x')## 添加x轴的名称
# plt.ylabel('y')## 添加y轴的名称
# plt.xlim([0,20])## 确定x轴范围
# plt.ylim([0,20])## 确定y轴范围
# plt.xticks([0,5,10,15,20])## 规定x轴刻度
# plt.yticks([0,5,10,15,20])## 确定y轴刻度
# plt.plot(X,optimal[0]+optimal[1]*X)
# plt.scatter(X1,y)
# plt.show()

# # def mse(b,w,points):
# #     #根据当前的w,b参数计算均方差损失
# #     totalError = 0
# #     for i in range(0,len(points)):#循环迭代所有点
# #         x = points[i,0]#获得i号点的输入x
# #         y = points[i,1]#获得i号点的输出y
# #         #计算差的平方，并累加
# #         totalError += (y - (w * x + b)) ** 2
# #     #将累加的误差求平均，得到均方差
# #     return totalError/float(len(points))


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import *
from mpl_toolkits import mplot3d #用于绘制3D图形
#梯度函数的导数
def gradJ1(theta):
    return 4*theta
def gradJ2(theta):
    return 2*theta
 
#梯度函数
def f(x, y):
    return  2*x**2 +y**2

def ff(x,y):
    return 2*np.power(x,2)+np.power(y,2)

def train(lr,epoch,theta1,theta2,up,dirc):
    t1 = [theta1]
    t2 = [theta2]
    for i in range(epoch):
        gradient = gradJ1(theta1)
        theta1 = theta1 - lr*gradient
        t1.append(theta1)
        gradient = gradJ2(theta2)
        theta2 = theta2 - lr*gradient
        t2.append(theta2)
        
    plt.figure(figsize=(10,10))     #设置画布大小
    x = np.linspace(-3,3,30)
    y = np.linspace(-3,3,30)
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)    
    ax = plt.axes(projection='3d')
    fig =plt.figure()
    #ax1 = plt.subplot(2, 1, 1)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none') #曲面图
    #ax.plot_wireframe(X, Y, Z, color='c') #线框图
    ax.contour3D(X, Y, Z, 50, cmap='binary')#等高线图
    #fig =plt.figure()
    #print(t1)
    #print(ff(t1,t2)+10)
    #ax1 = plt.subplot(2, 2, 1)
    ax.scatter3D(t1, t2, ff(t1,t2), c='r',marker = 'o')
    #ax.plot3D(t1, t2,  ff(t1,t2),'red')
    #调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    ax.view_init(up, dirc)

#可以随时调节，查看效果 (最小值，最大值，步长)
@interact(lr=(0, 2, 0.0002),epoch=(1,100,1),init_theta1=(-3,3,0.1),init_theta2=(-3,3,0.1),up=(-180,180,1),dirc=(-180,180,1),continuous_update=False)
#lr为学习率（步长） epoch为迭代次数   init_theta为初始参数的设置 up调整图片上下视角 dirc调整左右视角
def visualize_gradient_descent(lr=0.05,epoch=10,init_theta1=-2,init_theta2=-3,up=45,dirc=100):
    train(lr,epoch,init_theta1,init_theta2,up,dirc)
