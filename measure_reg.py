import pandas as pd
import os

path3=r'C:\data\measure'
df=pd.read_json(os.path.join(path3,'records.json'),orient='records')
print(df)