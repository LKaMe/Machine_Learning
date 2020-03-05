# from keras.models import Sequential 
# from keras.layers import Embedding,Flatten,Dense 

# model = Sequential()
# model.add(Embedding(max_words,embedding_dim,input_length = maxlen))
# model.add(Flatten())
# model.add(Dense(32,activation = 'relu'))
# model.add(Dense(1,activation = 'sigmoid'))
# model.summary()

# model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['acc'])
# history = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_data = (x_val,y_val))