import numpy as np 

samples = ['The cat sat on the mat.','The dog ate my homework.']#初始数据：每个样本是列表的一个元素(本例中的样本是一个句子，但也可以是一整篇文档)
token_index = {}#构建数据中所有标记的索引
for sample in samples:
    for word in sample.split():#利用split方法对样本进行分词，在实际应用中，还需要从样本中去掉标点和特殊字符
        if word not in token_index:
            token_index[word] = len(token_index) + 1 #为每个唯一单词指定一个唯一索引。注意，没有为索引编号0指定单词
max_length = 10#对样本进行分词。只考虑每个样本前max_length个单词
results = np.zeros(shape=(len(samples),max_length,max(token_index.values()) + 1)) #将结果保存在results中
for i,samples in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index] = 1