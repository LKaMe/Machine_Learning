for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batchs(data,batch_size = 50): #使用mini-batch 为50的样本集进行迭代
        params_grad = evaluate_gradient(loss_function,batch,params)
        params = params - learning_rate * params_grad 