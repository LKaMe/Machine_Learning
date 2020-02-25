# k = 4
# num_validation_samples = len(data) // k

# np.random.shuffle(data)

# validation_scores = []
# for fold in range(k):
#     validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]#选择验证数据区分
#     training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]#使用剩余数据作为训练数据。注意，+运算符是列表合并，不是求和

#     #创建一个全新的模型实例（未训练）
#     model = get_model()
#     model.train(training_data)
#     validation_score = model.evaluate(validation_data)
#     validation_scores.append(validation_score)

# validation_score = np.average(validation_scores)#最终验证分数：K折验证分数的平均值

# #在所有非测试数据上训练最终模型
# model = get_model()
# model.train(data)
# test_score = model.evaluate(test_data)
