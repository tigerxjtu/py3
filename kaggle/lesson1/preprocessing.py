import numpy as np
import pandas as pd

dfTrain=pd.read_csv("c:/tmp/kaggle/work1/train.csv")
dfTest=pd.read_csv("c:/tmp/kaggle/work1/test.csv")
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

dfTrain["index"] = np.arange(num_train)
dfTest["index"] = np.arange(num_test)