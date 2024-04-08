#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[2]:


# data = pd.read_csv('../../triple_cosineSNF(-1and1).csv')
# index = list(range(len(data)))
# random.shuffle(index)
# dataShuffled =  data.iloc[index]


# In[3]:


# dataShuffled.to_csv('../../triple_cosineSNF(-1and1)1.csv',index=False)


# In[4]:


data = pd.read_csv('../../triple_cosineSNF(-1and1)_42702.csv')


# In[5]:


(data.iloc[:,0:3]).to_csv('drugsPairs(-1_1).csv',header = False)


# In[6]:


drugsPairs = pd.read_csv('drugsPairs(-1_1).csv',header=None)
r, c = drugsPairs.shape
print(r, c)


# In[7]:


drugsPairs.head()


# In[8]:


42702/2


# In[11]:


drugsPairs


# In[12]:


indexes = []
bufpairs = []
for i in range(21351):
    candInd = np.random.randint(r)
#     if (len(indexes)< r-1):
    if (candInd not in indexes):
        indexes.append(candInd)
#         print(candInd)
    else:
        while(True):
            candInd = np.random.randint(r)
            if candInd not in indexes:
                break
        indexes.append(candInd)
#         print(candInd)
    di, dj = str(drugsPairs.iloc[candInd,1]), str(drugsPairs.iloc[candInd,2])
#     print(di,dj)
    try:
        bufpairs = drugsPairs[drugsPairs[1]==dj][drugsPairs[2]==di]
        indexes.append(int(bufpairs[0]))
    except:
        print(bufpairs[0],indexes[-1])
        del indexes[-1]
        
print(len(indexes))


# In[13]:


# drugsPairs.iloc[[20828,24725,33347,27018],:]


# In[14]:


# index = drugsPairs.values[:,0]
# random.shuffle(index)
# drugsPairsShuffled =  drugsPairs.iloc[index]


# In[15]:


checked = False
problems = []
i = 0
while(i +1 < len(indexes)):
#     if i + 1 == len(indexes):
    if (drugsPairs.iloc[indexes[i],1]==drugsPairs.iloc[indexes[i+1],2])and(drugsPairs.iloc[indexes[i],2]==drugsPairs.iloc[indexes[i+1],1]):
        i += 2

    else:        
        problems.append([i,drugsPairs.iloc[i,:]])
#         print([i,drugsPairs.iloc[i,:]])
        del indexes[i]
#         i += 1
#         if i + 1 == len(indexes):
#             problems.append(list(i,drugsPairs.iloc[i,:]))
#             indexes.remove(i)

len(problems)


# In[16]:


data.iloc[indexes,:].to_csv('../../triple42702_ShuffledPaired.csv',index=False)


# In[17]:


data.iloc[indexes,:].iloc[:38432,:].to_csv('../../tripleTrain42702.csv',index=False)


# In[18]:


data.iloc[indexes,:].iloc[38432:,:].to_csv('../../tripleTest42702.csv',index=False)


# In[19]:


# data.drop(indexes, inplace=True)
# data.iloc[:,1:].to_csv('../../tripleTest.csv',index=False)


# In[20]:


# data = pd .read_csv('../../triple42702_ShuffledPaired.csv')
# data.iloc[:38800,:].to_csv('../../tripleTrain42702.csv',index=False)
# data.iloc[38800:,:].to_csv('../../tripleTest42702.csv',index=False)


# In[21]:


# [20,drugsPairs.iloc[20,:]]


# In[22]:


# problems = pd.DataFrame(problems)
# problems.to_csv('problems.csv', index=False)


# In[23]:


# indexes = pd.DataFrame(indexes)
# indexes.to_csv('indexes.csv', index=False)


# In[24]:


# indexes = pd.read_csv('indexes21351.csv')

# indexes = list(indexes['0'])

# # print(indexes[0]==122943)


# In[25]:


# data10000 = []
# line1 = 0
# with open('../../saved F(triple_cosineSNF).csv', 'r') as fd:
    
#     for row ,line in enumerate(fd):
#         if row-1 in indexes:
# #             print(row)
#             data10000.append(line.split(','))
# #         line1 = line.split(',')
# #         break
# # line1


# In[26]:


# dataAll = data


# In[27]:


# data = []
# officer = []
# # index = indexes
# # random.shuffle(index)
# for i in range(len(dataAll['0'])):
#     if i in officer:
#         continue
#     data.append(dataAll.iloc[i,:])
#     officer.append(i)
#     bufpairs = dataAll[dataAll[str(0)]==dataAll.iloc[i,1]]
#     data.append(bufpairs[bufpairs[str(1)]==dataAll.iloc[i,0]])
# #     for j in index:
# #         if j in officer:
# #             continue
# # #         print('add')
# #         if dataAll.iloc[i,1] == dataAll.iloc[j,0] and dataAll.iloc[i,0] == dataAll.iloc[j,1]:
# #             data.append(dataAll.iloc[j,:])
#     officer.append(int(bufpairs.index[bufpairs[str(1)]==dataAll.iloc[i,0]][0]))
# data = pd.DataFrame(data)
# data.to_csv('../../triple_cosineSNF(-1and1)2.csv', index=False)


# In[ ]:




