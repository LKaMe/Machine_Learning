#间隔100个step打印一次训练误差
if step % 100 == 0:
    print(step,'loss:',float(loss))

if step % 500 == 0:#每500个batch后进行一次测试(验证)
    #evaluate/test

for x,y in test_db:#对测验集迭代一遍
    h1 = x @ w1 + b1#第一层
    h1 = tf.nn.relu(h1)#激活函数
    h2 = h1 @ w2 + b2 #第二层
    h2 = tf.nn.relu(h2)#激活函数
    out = h2 @ w3 + b3#输出层

pred = tf.argmax(out,axis = 1)#选取概率最大的类别
y = tf.argmax(y,axis = 1)#one-hot编码逆过程
correct = tf.equal(pred,y)#比较预测值与真实值
total_correct += tf.reduce_sum(tf.cast(correct,dtype = tf.int32)).numpy()
#计算正确率
print(step,'Evaluate Acc:',total_correct/total)
