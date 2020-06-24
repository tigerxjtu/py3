import numpy as np
seed = 7
np.random.seed(seed)

import pandas as pd
import pandas.core.algorithms as algos
import os
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.basemap import Basemap
# %matplotlib inline

datadir = r'C:\projects\python\data\dataguru'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

genders=gatrain['gender'].value_counts()
groups=gatrain['group'].value_counts()

genders.plot(kind='bar')
plt.show()

groups.plot(kind='bar')
plt.show()

phones=phone['phone_brand'].value_counts()
phones.plot(kind='bar')
plt.show()

apps = appevents['is_active'].value_counts()
apps.plot(kind='bar')
plt.show()