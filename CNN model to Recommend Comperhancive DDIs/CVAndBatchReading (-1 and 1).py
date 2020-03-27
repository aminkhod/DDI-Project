#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[2]:


data = pd.read_csv('../../triple_cosineSNF(-1and1)1.csv').iloc[:,1:]
# index = list(range(len(data)))
# random.shuffle(index)
# dataShuffled =  data.iloc[index]
# data.to_csv('../../triple_cosineSNF(-1and1)1.csv')


# In[3]:


# (data.iloc[:,:3]).to_csv('drugsPairs(-1_1).csv')


# In[4]:


# drugsPairs = pd.read_csv('drugsPairs(-1_1).csv')
# r, c = drugsPairs.shape
# print(r, c)


# In[5]:


# indexes = []
# for i in range(21351):
#     candInd = np.random.randint(r)
# #     if (len(indexes)< r-1):
#     if (candInd not in indexes):
#         indexes.append(candInd)
# #         print(candInd)
#     else:
#         while(True):
#             candInd = np.random.randint(r)
#             if candInd not in indexes:
#                 break
#         indexes.append(candInd)
# #         print(candInd)
#     di, dj = str(drugsPairs.iloc[candInd,1]), str(drugsPairs.iloc[candInd,2])
# #     print(di,dj)
#     bufpairs = drugsPairs[drugsPairs[str(1)]==dj]
#     indexes.append(int(bufpairs[bufpairs[str(2)]==di]['0']))

# print(len(indexes))


# In[6]:


# index = drugsPairs.values[:,0]
# random.shuffle(index)
# drugsPairsShuffled =  drugsPairs.iloc[index]


# In[7]:


# checked = False
# problems = []
# i = 0
# while(i +1 < len(indexes)):
# #     if i + 1 == len(indexes):
#     if (drugsPairs.iloc[indexes[i],1]==drugsPairs.iloc[indexes[i+1],2])and(drugsPairs.iloc[indexes[i],2]==drugsPairs.iloc[indexes[i+1],1]):
#         i += 2

#     else:        
#         problems.append([i,drugsPairs.iloc[i,:]])
# #         print([i,drugsPairs.iloc[i,:]])
#         del indexes[i]
# #         i += 1
# #         if i + 1 == len(indexes):
# #             problems.append(list(i,drugsPairs.iloc[i,:]))
# #             indexes.remove(i)

# len(problems)


# In[8]:


# [20,drugsPairs.iloc[20,:]]


# In[9]:


# problems = pd.DataFrame(problems)
# problems.to_csv('problems21351.csv', index=False)


# In[10]:


# indexes = pd.DataFrame(indexes)
# indexes.to_csv('indexes21351.csv', index=False)


# In[11]:


# indexes = pd.read_csv('indexes21351.csv')

# indexes = list(indexes['0'])

# # print(indexes[0]==122943)


# In[12]:


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


# In[13]:


dataAll = data


# In[ ]:


data = []
officer = []
# index = indexes
# random.shuffle(index)
for i in range(len(dataAll['0'])):
    if i in officer:
        continue
    data.append(dataAll.iloc[i,:])
    officer.append(i)
    bufpairs = dataAll[dataAll[str(0)]==dataAll.iloc[i,1]]
    data.append(bufpairs[bufpairs[str(1)]==dataAll.iloc[i,0]])
#     for j in index:
#         if j in officer:
#             continue
# #         print('add')
#         if dataAll.iloc[i,1] == dataAll.iloc[j,0] and dataAll.iloc[i,0] == dataAll.iloc[j,1]:
#             data.append(dataAll.iloc[j,:])
    officer.append(int(bufpairs.index[bufpairs[str(1)]==dataAll.iloc[i,0]][0]))
data = pd.DataFrame(data)
data.to_csv('../../triple_cosineSNF(-1and1)2.csv', index=False)


# In[ ]:




