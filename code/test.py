#coding=utf-8
#读取图片 返回图片某像素点的b，g，r值
import cv2
import numpy as np
# print('E:\\python\\Machine_Learning\\data\\微信图片_20200707170342.png')
# img=cv2.read('E:\\python\\Machine_Learning\\data\\微信图片_20200707170342.png')
# img = cv2.imread('E:/python/Machine_Learning/data/微信图片_20200707170342.jpg')
# print(img)



# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# def write_csv(path, data, way):
#     # 按传入方式 way 写入csv
#     if ["a+", "w+", 'r+'].__contains__(way) is False:
#         log(['当前传入方式错误：', way])
#         sys.exit()

#     if type(data) is list:
#         data = ','.join(str(v) for v in data)
#     elif type(data) is tuple:
#         data = ','.join(str(v) for v in data)
#     else:
#         data = str(data)
#     f = open(path, way,encoding='utf-8')
#     f.write(data + '\n')
#     f.close()
    
# I = mpimg.imread('E:\\python\\Machine_Learning\\data\\微信图片_20200707170342.png')
# print (I.shape)
# for x in range(I.shape[0]):   # 图片的高
#     for y in range(I.shape[1]):   # 图片的宽
# 		# px = I[x,y]
#         print([x,y])   
#         write_csv('E:\\python\\Machine_Learning\\data\\像素.csv',[x,y],'a+')
# plt.imshow(I)

import cv2
img = cv2.imread('E:\\python\\Machine_Learning\\data\\20200707170342.png')
 
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("1:",img[y][x][0])
        print("2:",img[y][x][1])
        print("3:",img[y][x][2])
 
        xy = "%d,%d" % (x, y)
 
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)

