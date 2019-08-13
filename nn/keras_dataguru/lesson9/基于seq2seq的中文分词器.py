
# coding: utf-8

# Sequence-to-Sequence模型  
# <center><img src="seq2seq.jpg" alt="FAO" width="500"></center> 
# 1.可用于机器翻译  
# 2.文章摘要  
# 3.对话机器人  
# 4.中文分词  
# ......

# In[1]:


import re
import numpy as np
import pandas as pd

# 
text = open('msr_train_10.txt').read()
text = text.split('\n')


# In[2]:

len(text)


# In[3]:

# {B:begin, M:middle, E:end, S:single}，分别代表每个状态代表的是该字在词语中的位置，
# B代表该字是词语中的起始字，M代表是词语中的中间字，E代表是词语中的结束字，S则代表是单字成词
text


# In[4]:

# 设置参数
# 词向量长度
word_size = 128
# 设置最长的一句话为32个字
maxlen = 32
# 批次大小
batch_size = 1024

# 根据符号分句
text = u''.join( text)
text = re.split(u'[，。！？、]/[bems]', text)


# In[5]:

len(text)


# In[6]:

text


# In[7]:

# 训练集数据
data = []
# 标签
label = []

# 得到所有的数据和标签
def get_data(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        # 返回数据和标签，0为数据，1为标签
        return list(s[:,0]), list(s[:,1])

for s in text:
    d = get_data(s)
    if d:
        data.append(d[0])
        label.append(d[1])


# In[8]:

test = re.findall('(.)/(.)', '你/s  只/b  有/e  把/s  事/b  情/e  做/b  好/e')
test


# In[9]:

# 定义一个dataframe存放数据和标签
d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
# 提取data长度小于等于maxlen的数据
d = d[d['data'].apply(len) <= maxlen]
# 重新排列index
d.index = range(len(d))


# In[10]:

d


# In[11]:

#统计所有字，给每个字编号
chars = [] 
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()


# In[12]:

chars


# In[13]:

chars[:] = range(1, len(chars)+1)
chars


# In[14]:

#生成适合模型输入的格式
from keras.utils import np_utils

# 定义标签所对应的编号
tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})

# # 把中文变成编号，再补0
# d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
# # 把标签变成编号，再补0
# d['y'] = d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))


def data_helper(x):
    x = list(chars[x]) + [0]*(maxlen-len(x))
    return np.array(x)     

def label_helper(x):
    x = list(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1))))
    x = x + [np.array([[0,0,0,0,1]])]*(maxlen-len(x))
    return np.array(x) 
    
d['x'] = d['data'].apply(data_helper) 
d['y'] = d['label'].apply(label_helper)    


# In[15]:

d['data'][0]


# In[16]:

d['x'][0]


# In[17]:

d['label'][0]


# In[18]:

d['y'][0]


# <center><img src="lstm1.png" alt="FAO" width="500"></center> 
# <center><img src="lstm2.png" alt="FAO" width="500"></center> 

# In[19]:

# 设计模型
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
from keras.models import load_model

sequence = Input(shape=(maxlen,), dtype='int32')
# 词汇数，词向量长度，输入的序列长度，是否忽略0值
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
# 双向RNN包装器
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
# 该包装器可以把一个层应用到输入的每一个时间步上
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
# 定义模型输出输出
model = Model(inputs=sequence, outputs=output)
# 定义代价函数，优化器
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# print(np.array(list(d['x'])).shape)
# print(np.array(list(d['y'])).reshape((-1,maxlen,5)).shape)
# model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, epochs=20)
# model.save('seq2seq.h5')

print("load model")
model = load_model('seq2seq.h5')


# In[20]:

model.summary()


# 做预测

# In[21]:

# 使用全数据
text = open('msr_train.txt').read()
text = text.split('\n')

# 根据符号分句
text = u''.join(text)
text = re.split(u'[，。！？、]/[bems]', text)

# 训练集数据
data = []
# 标签
label = []

# 得到所有的数据和标签
def get_data(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        # 返回数据和标签，0为数据，1为标签
        return list(s[:,0]), list(s[:,1])

for s in text:
    d = get_data(s)
    if d:
        data.append(d[0])
        label.append(d[1])
        
# 定义一个dataframe存放数据和标签
d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
# 提取data长度小于等于maxlen的数据
d = d[d['data'].apply(len) <= maxlen]
# 重新排列index
d.index = range(len(d))

#统计所有字，给每个字编号
chars = [] 
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)


# In[22]:

# 统计状态转移
dict_label = {}
for label in d['label']:
    for i in range(len(label)-1):
        tag = label[i] + label[i+1]
        dict_label[tag] = dict_label.get(tag,0) + 1
print(dict_label)


# In[23]:

# 计算状态转移总次数
sum_num = 0
for value in dict_label.values():
    sum_num = sum_num + value
sum_num


# In[24]:

# 计算状态转移概率
p_ss = dict_label['ss']/sum_num
p_sb = dict_label['sb']/sum_num
p_bm = dict_label['bm']/sum_num
p_be = dict_label['be']/sum_num
p_mm = dict_label['mm']/sum_num
p_me = dict_label['me']/sum_num
p_es = dict_label['es']/sum_num
p_eb = dict_label['eb']/sum_num


# In[25]:

# 维特比算法，维特比算法是一种动态规划算法用于寻找最有可能产生观测事件序列的-维特比路径

# tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})

# 00 = ss = 1
# 01 = sb = 1
# 02 = sm = 0
# 03 = se = 0
# 10 = bs = 0
# 11 = bb = 0
# 12 = bm = 1
# 13 = be = 1
# 20 = ms = 0
# 21 = mb = 0
# 22 = mm = 1
# 23 = me = 1
# 30 = es = 1
# 31 = eb = 1
# 32 = em = 0
# 33 = ee = 0

# 定义状态转移矩阵
transfer = [[p_ss,p_sb,0,0],
            [0,0,p_bm,p_be],
            [0,0,p_mm,p_me],
            [p_es,p_eb,0,0]]

# # 定义状态转移矩阵
# transfer = [[1,1,0,0],
#             [0,0,1,1],
#             [0,0,1,1],
#             [1,1,0,0]]

# 根据符号断句
cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')

# 预测分词
def predict(sentence):
    
    # 如果句子大于最大长度，只取maxlen个词
    if len(sentence) > maxlen:
        sentence = sentence[:maxlen]
    
    # 预测结果，先把句子编程编号的形式，如果出现生僻字就填充0，然后给句子补0直到maxlen的长度。预测得到的结果只保留跟句子有效数据相同的长度
    result = model.predict(np.array([list(chars[list(sentence)].fillna(0).astype(int))+[0]*(maxlen-len(sentence))]))[0][:len(sentence)]

    # 存放最终结果
    y = []
    # 存放临时概率值
    prob = []
    # 计算最大转移概率
    # 首先计算第1个词和第2个词,统计16种情况的概率
    # result[0][j]第1个词的标签概率
    # result[1][k]第2个词的标签概率
    # transfer[j][k]对应的转移概率矩阵的概率
    for j in range(4):
        for k in range(4):
            prob.append(result[0][j]*result[1][k]*transfer[j][k])
    # 计算前一个词的的标签
    word1 = np.argmax(prob)//4
    # 计算后一个词的标签
    word2 = np.argmax(prob)%4
    # 保存结果
    y.append(word1)
    y.append(word2)
    # 从第2个词开始
    for i in range(1,len(sentence)-1):
        # 存放临时概率值
        prob = []
        # 计算前一个词后后一个词的所有转移概率
        for j in range(4):
            prob.append(result[i][word2]*result[i+1][j]*transfer[word2][j])
        # 计算后一个词的标签
        word2 = np.argmax(prob)%4
        # 保存结果
        y.append(word2)
        
    # 分词
    words = []
    for i in range(len(sentence)):
        # 如果标签为s或b，append到结果的list中
        if y[i] in [0, 1]:
            words.append(sentence[i])
        else:
        # 如果标签为m或e，在list最后一个元素中追加内容
            words[-1] += sentence[i]
    return words

# 分词
def cut_word(s):
    result = []
    # 指针设置为0
    j = 0
    # 根据符号断句
    for i in cuts.finditer(s):
        # 对符号前的部分分词
        result.extend(predict(s[j:i.start()]))
        # 加入符号
        result.append(s[i.start():i.end()])
        # 移动指针到符号后面
        j = i.end()
    # 对最后的部分进行分词
    result.extend(predict(s[j:]))
    return result


# In[26]:

cut_word('基于seq2seq的中文分词器')


# In[27]:

cut_word('人们常说生活是一部教科书')


# In[28]:

cut_word('广义相对论是描写物质间引力相互作用的理论')


# In[29]:

model.predict(np.array([list(chars[list('今天天气很好')].fillna(0).astype(int))+[0]*(maxlen-len('今天天气很好'))]))[0]


# In[ ]:



