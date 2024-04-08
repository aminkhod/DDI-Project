#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[2]:


def reduceList( DrugPairs):
#     print(dPairs.head())
    i = 0
    newList = []
    seenList = []
    while i < len(DrugPairs.iloc[:,0]):
        d1, d2 = DrugPairs.iloc[i][1], DrugPairs.iloc[i][2]
        buf = DrugPairs[DrugPairs[1]==d2][DrugPairs[2]==d1]
#         print(buf)
        if len(buf) == 1 and (i not in seenList) and (list(buf.index)[0] not in seenList):
            newList.append(i)
            newList.append(list(buf.index)[0])
            seenList.append(list(buf.index)[0])
            seenList.append(i)
        else:
            seenList.append(i)
        i += 1
    return newList


# In[3]:


data = pd.read_csv('../../triple_cosineSNF(zeros).csv')


# In[ ]:


predict = pd.read_csv('predict_(-1 and +1 model)all.csv')
predict.head()


# In[ ]:


zeroList = list(predict[predict['0']<0.4][predict['1']<0.4].index)
len(zeroList)


# In[ ]:


probZeroDrug = data.iloc[zeroList,:]
probZeroDrug.head()
print(probZeroDrug.shape)


# In[ ]:


probZeroDrug.iloc[:,0:3].to_csv('probZeroDrugpairs.csv',header = False)
drugsPairs = pd.read_csv('probZeroDrugpairs.csv',header=None)
r, c = drugsPairs.shape
print(r, c)
drugsPairs.head()


# In[ ]:


newList = reduceList(drugsPairs.copy())
# print(newDegList)
len(newList)


# In[ ]:


probZeroDrug.iloc[newList,:].to_csv('probZeroDrugBothPair.csv',index=False)


# In[ ]:


probZeroDrugBothPair = pd.read_csv('probZeroDrugBothPair.csv')


# In[ ]:


drugsPairs = probZeroDrugBothPair.iloc[:,0:3]
r, c = drugsPairs.shape
print(r, c)
drugsPairs.head()


# In[ ]:


indexes = []
bufpairs = []
for i in range(int(15000)):
    candInd = np.random.randint(r/2)

    while (int(2 * candInd) in indexes):
        candInd = np.random.randint(r/2)
    indexes.append(int(2 * candInd))
    indexes.append(int(2 * candInd + 1))

print(len(indexes))


# In[ ]:


drugsPairs.iloc[indexes,:]


# In[ ]:


# index = drugsPairs.values[:,0]
# random.shuffle(index)
# drugsPairsShuffled =  drugsPairs.iloc[index]


# In[ ]:


checked = False
problems = []
i = 0
while(i +1 < len(indexes)):

    if (drugsPairs.iloc[indexes[i],0]==drugsPairs.iloc[indexes[i+1],1])and(drugsPairs.iloc[indexes[i],1]==drugsPairs.iloc[indexes[i+1],0]):
        i += 2

    else:        
        problems.append([i,drugsPairs.iloc[i,:]])

        del indexes[i]

len(problems)


# In[ ]:


probZeroDrug.iloc[indexes,:].to_csv('probZeroDrug30000.csv',index=False)


# In[ ]:


probZeroDrug.iloc[indexes,:].iloc[:18000,:].to_csv('probZeroDrugTrain.csv',index=False)


# In[ ]:


probZeroDrug.iloc[indexes,:].iloc[18000:,:].to_csv('probZeroDrugTest.csv',index=False)


# In[ ]:




