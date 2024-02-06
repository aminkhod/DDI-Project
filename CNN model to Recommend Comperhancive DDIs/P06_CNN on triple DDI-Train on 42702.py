#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import pydot
import pydotplus
import graphviz


# In[2]:


dataTrain = pd.read_csv('../../triple42702_ShuffledPaired.csv')
dataTest = pd.read_csv('../../tripleTest42702.csv')


# In[3]:


16*71


# In[4]:


X_train = dataTrain.values[:,3:]
y_train = dataTrain.values[:,2].astype(int)
del dataTrain
trainNum = len(X_train)
X_test = dataTest.values[:,3:]
y_test = dataTest.values[:,2].astype(int)
# del dataTest
testNum = len(X_test)


# In[5]:


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
# y_test[0]


# In[6]:


print(y_train[0:5], y_test[0:5])


# In[7]:


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
# model.add(Softmax(128))
model.summary()

#compile model using accuracy to measure model performance


adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
# model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist

### Load the model's saved weights.
# model.load_weights('cnn43110(1and-1)_rivised_8_epoch.h5')


# In[8]:


# ###plotting model
# plot_model(model,show_shapes = True, to_file='model.png')


# In[9]:


###### #### train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6)
# history = model.fit(X_train, y_train, epochs=1)


# In[10]:


### Saveing the Model
model.save_weights('Weight/CNN on triple DDI-Train on 42702_6_epoch.h5')


# In[ ]:




