# from keras.datasets import mnist
# import matplotlib.pyplot as plt
# (train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# # print(train_images.ndim)
# # print(train_images.shape)
# # print(train_images.dtype)
# digit = train_images[4]
# plt.imshow(digit,cmap = plt.cm.binary)
# # plt.show()
# my_slice = train_images[10:100]
# # print(my_slice.shape)

# #等同于上
# my_slice = train_images[10:100,:,:]
# # print(my_slice.shape)

# #也等同于上
# my_slice = train_images[10:100,0:28,0:28]
# # print(my_slice.shape)

# #逐元素运算