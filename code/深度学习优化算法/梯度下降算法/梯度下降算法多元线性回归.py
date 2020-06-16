import numpy as np
#定义多元线性模型
def model(theta,x):
    y_predict = np.dot(x,theta)
    return y_predict

#BGD算法
def bgd_algorithms(theta,x,y_real,alpha,e,iteration = 0):
    m = x.shape[0]
    while True:
        iteration += 1
        theta_gradient = 2/m * np.dot(x.T,(model(theta,x) - y_real))
        theta_norm = np.linalg.norm(theta_gradient)
        if theta_norm < e:
            break
        else:
            theta -= alpha * theta_gradient
        print('迭代%s次，损失量:%s,'%(iteration,theta_norm),end = '')
        print('模型参数:',end = '')
        for thetai in theta:
            print(thetai,end = '')
        print('\n')
    print('算法结束，迭代次数:'+str(iteration))
