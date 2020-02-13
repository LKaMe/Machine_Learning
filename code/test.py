import numpy as np 

x = np.array([[1],[2],[3]])
y = np.array([4,5,6])

#对y广播x
b = np.broadcast(x,y)
#它拥有iterator属性,基于自身组件的迭代器元组

print('对y广播x:')
r,c = b.iters

#python3.x为next(context),python2.x为context.next()
print(next(r),next(c))
print(next(r),next(c))
print('\n')
#shape属性返回广播对象的形状

print('广播对象的形状:')
print(b.shape)
print('\n')
#手动使用broadcast将x与y相加
b = np.broadcast(x,y)
c = np.empty(b.shape)

print('手动使用broadcast将x与y相加:')
print(c.shape)
print('\n')
c.flat = [u + v for (u,v) in b]

print('调用flat函数：')
print(c)
print('\n')
#获得了和numpy内建的广播支持相同的结果

print('x 与 y的和:')
print(x + y)