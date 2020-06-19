import numpy as np 
#其中p为LeakyReLU的负半段斜率，为超参数
def derivative(x,p):
    dx = np.ones_like(x) #创建梯度张量，全部初始化为1
    dx[x < 0] = p #元素为负的导数为p
    return dx