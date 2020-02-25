# # from keras import models
# # from keras import layers
# # from keras import optimizers
# # from keras import losses
# # from keras import metrics
# # model = models.Sequential()
# # model.add(layers.Dense(16,activation = 'relu',input_shape = (10000,)))
# # model.add(layers.Dense(16,activation = 'relu'))
# # model.add(layers.Dense(1,activation = 'sigmoid'))

# # model.compile(optimizer = optimizer.RMSprop(lr = 0.001),loss = 'binary_crossentropy',metrics = ['accuracy'])
# # model.compile(optimizer = optimizers.RMSprop(lr = 0.001),loss = losses.binary_crossentropy,metrics = [metrics.binary_accuracy])
# # x_val = x_train[:10000]
# # partical_x_train = x_train[10000:]

# # y_val = y_train[:10000]
# # partical_y_train = y_train[10000:]
# import matplotlib.pyplot as plt 

# history_dict = history.history 
# loss_value = history_dict['loss']
# val_loss_values = history_dict['val_loss']

# epochs = range(1,len(loss_values) + 1)

# plt.plot(epochs,loss_values,'bo',label = 'Training loss')
# plt.plot(epochs,val_loss_values,'b',label = 'Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.clf()
# acc= history_dict['acc']
# val_acc = history_dict['val_acc']

# plt.plot(epochs,acc,'bo',label = 'Training acc')
# plt.plot(epochs,val_acc,'b',label = 'Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()

# #从头开始训练一个模型