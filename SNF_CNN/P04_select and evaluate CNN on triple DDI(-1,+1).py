#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import confusion_matrix_pretty_print
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data

from sklearn.metrics import confusion_matrix,classification_report,precision_score, auc, precision_recall_curve, roc_curve

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Softmax, Dropout
from keras import optimizers
from keras import metrics as kmetr
from keras.utils import plot_model


# In[42]:


dataTrain = pd.read_csv('../../tripleTrain42702.csv')
dataTest = pd.read_csv('../../tripleTest42702.csv')


# In[43]:


16*71


# In[44]:


X_train = dataTrain.values[:,3:]
y_train = dataTrain.values[:,2]
del dataTrain
trainNum = len(X_train)
X_test = dataTest.values[:,3:]
y_test = dataTest.values[:,2]
# del dataTest
testNum = len(X_test)


# In[45]:


#reshape data to fit model
X_train = X_train.reshape(trainNum,16,71,1).astype('float32')
X_test = X_test.reshape(testNum,16,71,1).astype('float32')

y_train = y_train + 1
y_test  = y_test + 1
y_train = y_train / 2
y_test  = y_test / 2
print(y_train[0:5], y_test[0:5])

#one-hot encode target column
y_train = to_categorical(y_train).astype(int)
y_test = to_categorical(y_test).astype(int)
y_test[0]


# In[46]:


print(y_train[0:5], y_test[0:5])


# In[47]:


type(X_train[0][0][0][0])


# In[48]:


#create model
model = Sequential()
#add model layers
# kernel_initializer='uniform',
# kernel_initializer='uniform',
# kernel_initializer='uniform',
# kernel_initializer='uniform',
model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# model.add(Conv2D(64, kernel_size=2, activation='relu'))

model.add(Conv2D(32, kernel_size=4, activation='relu'))
# model.add(Conv2D(16, kernel_size=2, activation='relu'))
model.add(Conv2D(8, kernel_size=4, activation='relu'))
model.add(Flatten())
model.add(Dense( 64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense( 16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
# model.add(Softmax())
model.summary()

#compile model using accuracy to measure model performance


adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
# model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist

### Load the model's saved weights.
# model.load_weights('Weight/selected CNN on (-1, +1) DDI-Train of 42702Pair_5_epoch.h5')


# In[49]:


##plotting model
plot_model(model,show_shapes = True, to_file='model.png')


# In[50]:


# #### train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)
# history = model.fit(X_train, y_train, epochs=1)


# In[51]:


### Saveing the Model
model.save_weights('Weight/cnnSelection42702(1and-1)_15_epoch.h5')


# In[52]:


predit = model.predict(X_test)
#actual results for first 4 images in test set
print(predit[:4])


# In[53]:


# #from sklearn.metrics import precision_recall_curve, roc_curve

prec, rec, thr = precision_recall_curve(y_test[:,0], predit[:,0])
aupr_val = auc(rec, prec)
fpr, tpr, thr = roc_curve(y_test[:,0], predit[:,0])
auc_val = auc(fpr, tpr)
print(aupr_val,auc_val)


# In[1]:


# history.history


# In[60]:


# Plot training & validation accuracy values
plt.plot(list(range(1,16)),history.history['accuracy'])
plt.plot(list(range(1,16)),history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(list(range(1,16)),history.history['loss'])
plt.plot(list(range(1,16)),history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


# # predit
# predit[:,0].shape 


# In[61]:


predicts = []
for a,b in predit:
    if a >=b:
        predicts.append(0)
    else:
        predicts.append(1)
len(predicts)


# In[62]:


predicts1 = []
e = d = z = 0

for a,b in predit:
    if a >=0.90:
        predicts1.append(0)
        d += 1
    elif b>=0.95:
        predicts1.append(2)
        e += 1
    elif a<=0.05 and b<=0.1:
        predicts1.append(1)
        z += 1
print('degrassive', d, 'enhancive', e, 'zeros', z)
print("""
Epoch04: degrassive 224 enhancive 2939 zeros 40
Epoch05: degrassive 280 enhancive 2823 zeros 39
Epoch06: degrassive 233 enhancive 2879 zeros 79
Epoch07: degrassive 203 enhancive 2926 zeros 134
Epoch08: degrassive 224 enhancive 2895 zeros 180
Epoch09: degrassive 191 enhancive 2856 zeros 191
Epoch10: degrassive 189 enhancive 2821 zeros 246
Epoch11: degrassive 164 enhancive 2581 zeros 235
Epoch12: degrassive 166 enhancive 2454 zeros 266
""")


# In[63]:


# max(list((dataTest.values[:,2]+1)/2))


# In[64]:


cm = confusion_matrix(list(predicts), list((dataTest.values[:,2]+1)/2))
print(cm)

CR = classification_report(list((dataTest.values[:,2]+1)/2),list(predicts))
print(CR)
# print(145/4702)
# i=0
# for j in list(data.values[9500:,2]+1):
#     if j==1:
#         i +=1
# print(i)

# plt.show()
plot_confusion_matrix_from_data(list((dataTest.values[:,2]+1)/2), list(predicts))


# In[65]:


print(pd.DataFrame(predit))


# In[66]:


pd.DataFrame(predit).plot.density()


# In[67]:


pd.DataFrame(predit).iloc[:,0].plot.density()


# In[68]:


pd.DataFrame(predit).iloc[:,1].plot.density()


# In[69]:


fig, ax = plt.subplots()
fig.set_size_inches(16, 8)

# matplotlib histogram
# plt.hist(pd.DataFrame(predit).iloc[:,1], color = 'blue', edgecolor = 'black',
#          bins = int(200))

# seaborn histogram
sns.distplot(pd.DataFrame(predit).iloc[:,1], hist=True, kde=False, 
             bins=int(100), color = 'blue',
             hist_kws={'edgecolor':'black'})

# sns.distplot(pd.DataFrame(predit).iloc[:,0], hist=True, kde=True, 
#              bins=int(200), color = 'darkblue', 
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})
# Add labels
plt.title('frequency Histogram of Drugs')
plt.xlabel('Enhancive drugs Probability')
plt.ylabel('frequency distribution')


# In[70]:


fig, ax = plt.subplots()
fig.set_size_inches(16,8)

# matplotlib histogram
# plt.hist(pd.DataFrame(predit).iloc[:,1], color = 'blue', edgecolor = 'black',
#          bins = int(200))

# seaborn histogram

sns.distplot(pd.DataFrame(predit).iloc[:,0], hist=True, kde=False, 
             bins=int(100), color = 'red',
             hist_kws={'edgecolor':'black'})
# sns.distplot(pd.DataFrame(predit).iloc[:,0], hist=True, kde=True, 
#              bins=int(200), color = 'darkblue', 
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})
# Add labels
plt.title('frequency Histogram of Drugs')
plt.xlabel('Degressive drugs Probability')
plt.ylabel('frequency distribution')


# In[71]:


fig, ax = plt.subplots()
fig.set_size_inches(16,8)

# matplotlib histogram
# plt.hist(pd.DataFrame(predit).iloc[:,1], color = 'blue', edgecolor = 'black',
#          bins = int(200))

# seaborn histogram
sns.distplot(pd.DataFrame(predit).iloc[:,1], hist=True, kde=False, 
             bins=int(100), color = 'blue',
             hist_kws={'edgecolor':'black'})

sns.distplot(pd.DataFrame(predit).iloc[:,0], hist=True, kde=False, 
             bins=int(100), color = 'red',
             hist_kws={'edgecolor':'black'})
# sns.distplot(pd.DataFrame(predit).iloc[:,0], hist=True, kde=True, 
#              bins=int(200), color = 'darkblue', 
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})
# Add labels
plt.title('frequency Histogram of Drugs')
plt.xlabel('both of Degressive and Enhancive drugs Probability')
plt.ylabel('frequency distribution')


# In[ ]:




