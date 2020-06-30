import tensorflow as tf 

def preprocess(x,y):
    #预处理函数
    #x:图片的路径，y:图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x,channels=3)#RGBA
    #图片缩放到244*244大小，这个大小根据网络设定自行调整
    x = tf.image.resize(x,[244*244])

#旋转
#图片逆时针旋转180度
x = tf.image.rot90(x,2)

#翻转
#随机水平翻转
x = tf.image.random_flip_left_right(x)
#随机竖直翻转
x = tf.image.random_flip_up_down(x)

#裁剪
#图片先缩放到稍大尺寸
x = tf.image,resize(x,[244,244])
#再随机裁剪到合适尺寸
x = tf.image_crop(x,[224,224,3])
