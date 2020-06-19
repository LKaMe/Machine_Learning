import numpy as np 
def derivative(x):
    d = np.array(x,copy = True)#用来保存梯度的张量
    d[x < 0] = 0#元素为负的导数为0
    d[x >= 0] = 1#元素为正的导数为1
    return d
