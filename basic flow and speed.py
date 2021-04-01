# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:01:34 2021

@author: panyy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import norm
import seaborn as sns
from matplotlib.font_manager import FontProperties

df = pd.read_csv('CA_I405_bottleneck_13.51_train.csv')
print(df.describe())

date_1=df[df.date_id==1]
date_flow_1=np.array(date_1['Flow per hour'])
plt.figure()
plt.plot(date_flow_1)
plt.show()

date_speed_1=np.array(date_1['Speed'])
plt.figure()
plt.plot(date_speed_1)
plt.show()

x=np.array(date_1['Flow per hour'])
y=np.array(date_1['Speed'])
plt.scatter(x,y,edgecolors='r',color='r',label ='outer layer',zorder=30)

for i in range(26):
    date_flow_20=df[df.date_id==i+1]
    date_flow_i=np.array(date_flow_20['Flow per hour'])
    plt.figure()
    plt.plot(date_flow_i,'r',label='flow')
    plt.savefig('Flow.jpg')
    plt.show()  

for i in range(26):
    date_speed_20=df[df.date_id==i+1]
    date_speed_i=np.array(date_speed_20['Speed'])
    plt.figure()
    plt.plot(date_speed_i,'r',label='speed')
    plt.show()
    
for i in range(26):
    date_flow_speed_20=df[df.date_id==i+1]
    flow_speed=date_flow_speed_20.loc[:,['Flow per hour','Speed']]
    flow_speed.plot(x='Flow per hour',y='Speed',kind='scatter')

plt.figure(figsize=(16,8),dpi=600)
plt.ylabel('frequency')
plt.title('flow per hour')
plt.hist(x = df['Flow per hour'],  bins = 20, color = 'steelblue', edgecolor = 'black')

x=df['Flow per hour']
sns.displot(x,kde=True)
plt.ylabel('frequency')
plt.title('flow per hour')














