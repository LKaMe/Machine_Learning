#梯度下降算法
# from numpy import *

# # y = mx + b
# # m is slope, b is y-intercept
# def compute_error_for_line_given_points(b, m, points):
#     totalError = 0
#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         totalError += (y - (m * x + b)) ** 2
#     return totalError / float(len(points))

# def step_gradient(b_current, m_current, points, learningRate):
#     b_gradient = 0
#     m_gradient = 0
#     N = float(len(points))
#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
#         m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
#     new_b = b_current - (learningRate * b_gradient)
#     new_m = m_current - (learningRate * m_gradient)
#     return [new_b, new_m]

# def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
#     b = starting_b
#     m = starting_m
#     for i in range(num_iterations):
#         b, m = step_gradient(b, m, array(points), learning_rate)
#     return [b, m]

# def run():
#     points = genfromtxt("data/data.csv", delimiter=",")
#     learning_rate = 0.0001
#     initial_b = 0 # initial y-intercept guess
#     initial_m = 0 # initial slope guess
#     num_iterations = 1000
#     print ("Starting gradient descent at b = ",initial_b, "m = ",initial_m, "error = ", compute_error_for_line_given_points(initial_b, initial_m, points))
#     print ("Running...")
#     [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
#     print ("After" ,num_iterations,"iterations b = ",b, "m = ",m, "error = ", compute_error_for_line_given_points(b, m, points))

# if __name__ == '__main__':
#     run()

# import numpy as np 
# import matplotlib.pyplot as plt 
# #计算正弦曲线上点的x和y坐标
# x = np.arange(0,3*np.pi,0.1)
# y = np.sin(x)
# plt.title("sine wave from")
# #使用matplotlib来绘制点
# plt.plot(x,y)
# plt.show()