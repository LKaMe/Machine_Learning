import tensorflow as tf 
#模型端
#创建监控类，监控数据将写入log_dir目录
summary_writer = tf.summary.create_file_writer(log_dir)

with summary_writer.as_default():#写入环境
    #当前时间戳step上的数据为loss,写入到名为train-loss数据库中
    tf.summary.scalar('train-loss',float(loss),step=step)

with summary_writer.as_default():#写入环境
    #写入测试准确率
    tf.summary.scalar('test-acc',float(total_correct/total),step=step)
    #可视化测试用的图片，设置最多可视化9张图片
    tf.summary.image("val-onebyone-images:",val_images,max_outputs=9,step=step)

with summary_writer.as_default():
    #当前时间戳step上的数据为loss,写入到ID位train-loss对象中
    tf.summary.scalar('train-loss',float(loss),step=step)
    #可视化真实标签的直方图分布
    tf.summary.histogram('y-hist',y,step=step)
    #查看文本信息
    tf.summary.text('loss-text',str(float(loss)))