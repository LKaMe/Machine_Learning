import numpy as np 
def sigmoid(x):#实现sigmoid函数
    return 1/(1+np.exp(-x))

def derivative(x):#sigmoid导数的计算
    #sigmoid函数的表达式由手动推导而得
    return sigmoid(x)*(1-sigmoid(x))
    