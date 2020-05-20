
for i in range(nb_epochs): #nb_epochs为预先定义好的迭代次数
    params_grad = evaluate_gradient(loss_function,data,params) #
    params = params - learning_rate * params_grad 