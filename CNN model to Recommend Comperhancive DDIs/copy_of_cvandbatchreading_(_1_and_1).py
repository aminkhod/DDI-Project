# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import random
import os

data = pd.read_csv('triple_cosineSNF(-1and1)1.csv').iloc[:,1:]

dataAll = data
del data
indexes = pd.read_csv('triple_cosineSNF(-1and1)1.csv').iloc[:,0]

data = []
officer = []
index = indexes
random.shuffle(index)
for i in range(18000):
    if i in officer:
        continue
    data.append(dataAll.iloc[i,:])
    officer.append(i)
    # bufpairs = dataAll[dataAll[str(0)]==dataAll.iloc[i,1]]
    # data.append(bufpairs[bufpairs[str(1)]==dataAll.iloc[i,0]])
    for j in index:
        if j in officer:
            continue
#         print('add')
        if dataAll.iloc[i,1] == dataAll.iloc[j,0] and dataAll.iloc[i,0] == dataAll.iloc[j,1]:
            data.append(dataAll.iloc[j,:])
            officer.append(j)
            break
    # officer.append(int(bufpairs.index[bufpairs[str(1)]==dataAll.iloc[i,0]][0]))
            
data = pd.DataFrame(data)
data.to_csv('tripleTrain.csv', index=False)
dataAll.drop(officer, inplace=True)
dataAll.to_csv('tripleTest.csv', index=False)
