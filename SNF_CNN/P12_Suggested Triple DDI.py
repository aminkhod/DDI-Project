#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[14]:


pd.read_csv('triple_cosineSNF(zeros)_rivised.csv').iloc[:,0:2].to_csv('zero drug pairs.csv', index = False)


# In[25]:


zeroDrugPairs = pd.read_csv('zero drug pairs.csv')
zeroDrugPairs = zeroDrugPairs.rename(columns={'0': "i", "1": "j"})

zeroDrugPairs.head()


# In[32]:


predict = pd.read_excel('evi_predict_all_without softmax.xlsx')
predict = predict.rename(columns={0: "pMines1", 1: "p0",2:'p1'})
deg = predict["pMines1"]*(1-predict['p1'])
enh = predict['p1']*(1-predict["pMines1"])
predict = predict.join(zeroDrugPairs).join(pd.DataFrame({'deg':deg})).join(pd.DataFrame({'enh':enh}))

predict.head()


# In[33]:


def reduceList(pairList, predictCopy):
    dPairs = predictCopy.iloc[pairList,:]
    print(dPairs.head())
    i = 0
    newList = []
    seenList = []
    while i < len(pairList):
        d1, d2 = dPairs.iloc[i][3], dPairs.iloc[i][4]
        buf = dPairs[dPairs['i']==d2][dPairs['j']==d1]
#         print(buf)
        if len(buf) == 1 and (pairList[i] not in seenList) and (list(buf.index)[0] not in seenList):
            newList.append(pairList[i])
            newList.append(list(buf.index)[0])
            seenList.append(list(buf.index)[0])
            seenList.append(pairList[i])
        else:
            seenList.append(pairList[i])
        i += 1
    return newList, predictCopy.iloc[newList]


# In[34]:


# predict = predict.sort_values(by=['0'], ascending=False)
# # predict.head()
# degList = list(predict[predict['0']>0.9][predict['2']<0.01].index)
# len(degList)
# predict = predict.sort_values(by=['deg'], ascending=False).reindex(range(len(predict.iloc[:,0])))
# print(predict.head())
degList = list(predict[predict['deg']>0.99][predict['p0']<0.1].index)
len(degList)


# In[213]:


# dPairs = zeroDrugPairs.iloc[degList]
# dPairs.iloc[0]


# In[214]:


# i = 150
# print(zeroDrugPairs.iloc[i])
# d1, d2 = zeroDrugPairs.iloc[i][0], zeroDrugPairs.iloc[i][1]
# buf = zeroDrugPairs[zeroDrugPairs['0']==d2][zeroDrugPairs['1']==d1]
# # buf = buf[buf['1']==d1]
# buf
# # list(buf.index)[0]
# print(zeroDrugPairs.iloc[3])


# In[36]:


newDegList, newDegData = reduceList(degList.copy(), predict.copy())
# print(newDegList)
newDegData.shape


# In[205]:


352/2


# In[37]:


dropList = list(range(0, newDegData.shape[0], 2))
newDegData.iloc[dropList,:].to_csv('suggested Degressive.csv', index=False)


# In[38]:


# predict = pd.read_excel('predict_allTrain_epoch10.xlsx')


# In[40]:


# predict = predict.sort_values(by=['2'], ascending=False)
# enhList = list(predict[predict['2']>0.99][predict['0']<0.05].index)
# len(enhList)
# predict = predict.sort_values(by=['inh'], ascending=False).reindex(range(len(predict.iloc[:,0])))
# print(predict.head())
enhList = list(predict[predict['enh']>0.99][predict['p0']<0.1].index)
len(enhList)


# In[42]:


newEnhList, newenhData = reduceList(enhList.copy(), predict.copy())
# print(newEnhList)
newenhData.shape


# In[43]:


dropList = list(range(1, newenhData.shape[0], 2))
newenhData.iloc[dropList,:].to_csv('suggested Enhancive.csv', index=False)


# In[44]:


predict.iloc[[159189]]


# In[220]:


4476/2


# In[ ]:




