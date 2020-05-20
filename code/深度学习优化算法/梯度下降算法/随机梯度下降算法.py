import numpy as np 
for i in range(nb_epochs):
    np.random.shuffle(data)#每次迭代的时候打乱训练集
    for example in data:
        params_grad = evaluate_gradient(loss_function,example,params)
        params = params - learning_rate * params_grad

