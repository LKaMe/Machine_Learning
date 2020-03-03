# from keras.model import Sequential
# from keras.layers import Flatten,Dense,Embedding 

# model = Sequential()
# model.add(Embedding(10000,8,input_length=maxlen))#指定Embedding层的最大输入长度，以便后面将嵌入输入展平，Embedding层激活的形状为(samples,maxlen,8)

# model.add(Flatten())#将三维的嵌入张量展平成形状为(samples,maxlen*8)的二维张量

# model.add(Dense(1,activation='sigmoid'))#在上面添加分类器
# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
# model.summary()

# history = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_split = 0.2)