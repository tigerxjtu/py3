import pandas as pd
import numpy as np
import nltk


train=pd.read_csv("c:/tmp/kaggle/work1/train.csv")
test=pd.read_csv("c:/tmp/kaggle/work1/test.csv")

train.head()
test.head()

print(train.columns)
print(test.columns)

print(len(train))
print(len(test))

train['question1'].unique()[0:10]

len(train['question1'].unique())

len(test['question1'].unique())

train['question2'].unique()[0:10]

len(train['question2'].unique())

len(test['question2'].unique())

len(np.setdiff1d(test['question1'].unique(),train['question1'].unique()))

len(np.intersect1d(test['question1'].unique(),train['question1'].unique()))

train.fillna("",inplace=True)
question1 = train['question1'].map(nltk.tokenize.word_tokenize)
test.fillna("",inplace=True)
question2 = train['question2'].map(nltk.tokenize.word_tokenize)

from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
import matplotlib.pyplot as plt

def key_plot(data,col,top_num=10):
    s= data[col].map(nltk.tokenize.word_tokenize)
    fdist=nltk.FreqDist( words.lower()  for x in s
                    for words in x if words.lower() not in stopset )
    top=pd.DataFrame(fdist.most_common(top_num),columns=['question1','times'])
    # top=top.set_index('question1')
    plt.bar(top['question1'],top['times'])
    plt.show()
    # top.plot(kind='bar')


key_plot(train,'question1')
key_plot(test,'question1')