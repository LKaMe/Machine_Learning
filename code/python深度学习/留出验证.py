# num_validation_samples = 10000
# np.random.shuffle(data)#通常需要打乱数据

# validation_data = data[:num_validation_samples]
# data = data[num_validation_samples:]

# training_data = data[:]#定义训练集

# model = get_model()
# #在训练数据上训练模型，并在验证数据上评估模型
# model.train(training_data)
# validation_score = model.evaluate(validation_data)

# #现在你可以调节模型、重新训练、评估、然后再次调节
# model = get_model()
# model.train(np.concatenate([training_data,validation_data]))
# test_score = model.evaluate(test_data)
