import numpy as np 

def sigmoid(x):#sigmoid函数实现
    return 1/(1+np.exp(-x))

def tanh(x):#tanh函数实现
    return 2 * sigmoid(2 * x) - 1

def derivative(x):#tanh导数实现
    return 1-tanh(x) ** 2