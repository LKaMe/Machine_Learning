import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import *
from mpl_toolkits.mplot3d import Axes3D #用于绘制3D图形

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

def train(eta,epoch,theta1,theta2,up,dirc):
    t1 = [theta1]
    t2 = [theta2]
    for i in range(epoch):
        gradient = gradJ1(theta1)
        theta1 = theta1 - eta*gradient
        t1.append(theta1)
        gradient = gradJ2(theta2)
        theta2 = theta2 - eta*gradient
        t2.append(theta2)
        
    x = np.linspace(-3,3,30)
    y = np.linspace(-3,3,30)
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)    
    fig =plt.figure()
    ax = Axes3D(fig)

    # ax.contour3D(X, Y, Z, 50, cmap='binary')#等高线图
    ax.contour3D(X,Y,Z,50,cmap='rainbow')#彩色
    ax.scatter3D(t1, t2, ff(t1,t2), c='r',marker = 'o')
    ax.view_init(up, dirc)
    plt.rcParams['font.sans-serif'] = 'SimHei'#用于显示中文
    plt.rcParams['axes.unicode_minus'] = False #设置正常显示负号
    plt.title('Gradient Descent')
    plt.show()
# #可以随时调节，查看效果 (最小值，最大值，步长)
# @interact(eta=(0, 2, 0.0002),epoch=(1,100,1),init_theta1=(-3,3,0.1),init_theta2=(-3,3,0.1),up=(-180,180,1),dirc=(-180,180,1),continuous_update=False)
# #eta为学习率（步长） epoch为迭代次数   init_theta为初始参数的设置 up调整图片上下视角 dirc调整左右视角
# def visualize_gradient_descent(eta=0.05,epoch=10,init_theta1=-2,init_theta2=-3,up=45,dirc=100):
#     train(eta,epoch,init_theta1,init_theta2,up,dirc)
if __name__ == "__main__":
    train(0.05,10,-2,-3,45,100)