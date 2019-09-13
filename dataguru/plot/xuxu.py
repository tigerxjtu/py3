#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: XuXu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normfrom matplotlib.patches import Wedge as wedge
from matplotlib.gridspec import GridSpec as gsp
from matplotlib.font_manager import FontProperties
msyh = FontProperties(fname=r'C:\Users\A\Desktop\pyProject\msyh.ttf', size=15)
d = pd.read_csv(r'C:/Users/A/Desktop/pyProject/pyVis/w02/titanic_train.csv')
d['Fare'].dropna(inplace=True)
city = '登船城市:' + d.Embarked
city_table = city.value_counts()
pop = city_table / np.sum(city_table)
ang = pop * 360
ang = - np.cumsum(ang) + 90
ang = np.append(90, ang)
fare = d.Fare
fare_log = np.log(fare + 1)
my_color = ['#FF6600', '#EDC200', '#00BAC7', '#00BA38']
sns.set_style('darkgrid')
plt.figure(figsize=(12, 7))
plt.subplots_adjust(left=0.08, right=0.97, wspace=2, hspace=2)
gs = gsp(4, 7)#############################################################
plt.subplot(gs[:, :-3])
plt.bar(
np.arange(1, len(city_table)+1)-0.25,
np.repeat(city.count() / 500, len(city_table.index)),
color='black', alpha=0.1, edgecolor='k', linewidth=3, width=0.5
)
plt.bar(
np.arange(1, len(city_table) + 1) - 0.25, city_table / 500,
color=my_color[:-1], alpha=0.9, linewidth=0, width=0.5
)
plt.xlim([0.5, 3.5])
plt.ylim([0, 3])
plt.xticks([1, 2, 3], city_table.index, fontproperties=msyh)
plt.yticks([0, 0.5, 1, 1.5, 2], ['0', '250', '500', '750', '1 K'])
ax = plt.gca()
for x_temp in np.arange(1, 3+1):
plt.text(
x_temp, 2.4,
('%.1f' % (pop[x_temp - 1] * 100)) + ' %',ha='center', va='center', weight='bold',
fontsize=18, color=my_color[x_temp - 1]
)
if x_temp == 1:
color_txt = 'w'
else:
color_txt = my_color[x_temp - 1]
plt.text(
x_temp, 1.0, 'Top' + str(x_temp) + '\n' + str(city_table[x_temp - 1]),
ha='center', va='center', color=color_txt, fontsize=18, weight='bold'
)
for i in range(len(ang)-1):
w = wedge(
(x_temp, 2.4), 0.35, ang[i+1], ang[i],
width=0.1, color=my_color[i], alpha=0.25, linewidth=1
)
ax.add_patch(w)
w = wedge(
(x_temp, 2.4), 0.35, ang[x_temp], ang[x_temp - 1],
width=0.1, color=my_color[x_temp - 1], linewidth=0
)
ax.add_patch(w)plt.tick_params(top=False, right=False)
plt.ylabel('总人数', fontproperties=msyh, labelpad=10)
plt.title(
'各登船城市的人数分布图', family='SimHei',
fontsize=18, verticalalignment='bottom'
)
#############################################################
plt.subplot(gs[0:2, -3:])
sns.set_style('darkgrid')
sns.distplot(
fare, fit=norm, rug=True, bins=25,
rug_kws={'color': '#31A354'},
kde_kws={'color': '#006D2C', 'lw': 2, 'label': 'KDE'},
fit_kws={'color': '#232323', 'lw': 2, 'ls': '--'},
hist_kws={
'color': '#74C476',
'label': 'Hist'
}
)plt.annotate(
'偏离正态分布', fontproperties=msyh,
color='#FF6600', xy=(100, 0.004), xytext=(150, 0.012),
arrowprops=dict(
arrowstyle='->',
connectionstyle='arc3, rad=0.3',
color='#FF6600'
)
)
plt.annotate(
'偏离正态分布', fontproperties=msyh,
color='#FF6600', xy=(30, 0.014), xytext=(150, 0.012),
arrowprops=dict(
arrowstyle='->',
connectionstyle='arc3, rad=0.3',
color='#FF6600'
)
)
plt.xlim([-50, 300])
plt.ylim([0, 0.04])
plt.xticks(np.linspace(-50, 300, 8))
plt.yticks(np.linspace(0, 0.04, 5))plt.xlabel('船票收费', fontproperties=msyh)
plt.ylabel('概率密度', fontproperties=msyh, labelpad=10)
plt.title(
'船票收费的分布（对数化前） ', family='SimHei',
fontsize=16, verticalalignment='bottom'
)
plt.tick_params(top=False, right=False)
#############################################################
plt.subplot(gs[2:, -3:])
sns.set_style('darkgrid')
sns.distplot(
fare_log, fit=norm, rug=True, bins=15,
rug_kws={'color': '#3182BD'},
kde_kws={'color': '#08519C', 'lw': 2, 'label': 'KDE'},
fit_kws={'color': '#232323', 'lw': 2, 'ls': '--'},
hist_kws={
'color': '#6BAED6',
'label': 'Hist'
}
)plt.annotate(
'接近正态分布', fontproperties=msyh,
color='#FF6600', xy=(3.8, 0.30), xytext=(4.5, 0.35),
arrowprops=dict(
arrowstyle='->',
connectionstyle='arc3, rad=0.3',
color='#FF6600'
)
)
plt.annotate(
'接近正态分布', fontproperties=msyh,
color='#FF6600', xy=(2.6, 0.45), xytext=(4.5, 0.35),
arrowprops=dict(
arrowstyle='->',
connectionstyle='arc3, rad=0.3',
color='#FF6600'
)
)
plt.xlim([0, 7])
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 5))
plt.xlabel('对数化船票收费', fontproperties=msyh)