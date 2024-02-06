#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np

import seaborn as sn
import confusion_matrix_pretty_print
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,classification_report,precision_score
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Softmax, Dropout


# In[2]:


tripleData = pd.read_csv('../../triple42702_ShuffledPaired.csv')


# In[3]:


fold = 10 
interval = int(42702/fold)
auprListpositive1 = []
aucListpositive1 = []

auprListmines1 = []
aucListmines1 = []


# In[13]:


for split1 in range(10):
    split2 = split1 + 1
    
    trainIndex = list(range(0,split1*interval))
    trainIndex.extend(list(range(split2*interval,42702)))

    if split1==9:
        testIndex = list(range(split1*interval,42702))
    else:
        testIndex = list(range(split1*interval,split2*interval))

    dataTrain = tripleData.iloc[trainIndex,:]
    dataTest = tripleData.iloc[testIndex,:]
    print(min(testIndex),max(testIndex))


    X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
    y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
    trainNum = len(X_train)
    testNum = len(X_test)

    #reshape data to fit model
    X_train = X_train.reshape(trainNum,16,71,1).astype('float32')
    X_test = X_test.reshape(testNum,16,71,1).astype('float32')

    y_train = y_train + 1
    y_test  = y_test + 1
    y_train = y_train / 2
    y_test  = y_test / 2
    # print(y_train[0], y_test[0])

    from keras.utils import to_categorical
    #one-hot encode target column
    y_train = to_categorical(y_train).astype(int)
    y_test = to_categorical(y_test).astype(int)
    # y_test[0]
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
    model.add(Dense( 1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense( 64, activation='relu'))
    model.add(Dense( 2, activation='relu'))
    # model.add(Softmax())
    model.summary()


    #compile model using accuracy to measure model performance
    from keras import optimizers
    from keras import metrics as kmetr


    adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


    # # Load the model's saved weights.
    # model.load_weights('cnn.h5')

    #train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


    # Saveing the Model
    model.save_weights('Weight/cnn_'+str(split2)+'.h5')


    #predict first 4 images in the test set
    predit = model.predict(X_test)

    #actual results for first 4 images in test set
    print(predit[:4])


    #from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    prec, rec, thr = precision_recall_curve(y_test[:,0], predit[:,0])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_test[:,0], predit[:,0])
    auc_val = auc(fpr, tpr)
    aucListmines1.append(auc_val)
    auprListmines1.append(aupr_val)
    print(aupr_val,auc_val)

    #from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    prec, rec, thr = precision_recall_curve(y_test[:,1], predit[:,1])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_test[:,1], predit[:,1])
    auc_val = auc(fpr, tpr)
    aucListpositive1.append(auc_val)
    auprListpositive1.append(aupr_val)
    print(aupr_val , auc_val )


# In[14]:


from scipy.stats import sem, t
from scipy import mean
confidence = 0.95
# data = [1, 2, 3, 4, 5]

naucList = len(aucListpositive1)
maucList = mean(aucListpositive1)
std_erraucList = sem(aucListpositive1)
haucList = std_erraucList * t.ppf((1 + confidence) / 2, naucList - 1)


nauprList = len(auprListpositive1)
mauprList = mean(auprListpositive1)
std_errauprList = sem(auprListpositive1)
hauprList = std_errauprList * t.ppf((1 + confidence) / 2, nauprList  - 1)



print("positive1: "+ str(maucList) + '  -+' + str(haucList), str(mauprList) + '  -+' + str(hauprList))


# In[15]:


mypd = []
mypd = pd.DataFrame(mypd)
mypd['AUC'],mypd['AUPR'] = aucListpositive1, auprListpositive1
mypd.to_csv('selectedModelresult(+1).csv')


# In[16]:


from scipy.stats import sem, t
from scipy import mean
confidence = 0.95
# data = [1, 2, 3, 4, 5]

naucList = len(aucListmines1)
maucList = mean(aucListmines1)
std_erraucList = sem(aucListmines1)
haucList = std_erraucList * t.ppf((1 + confidence) / 2, naucList - 1)


nauprList = len(auprListmines1)
mauprList = mean(auprListmines1)
std_errauprList = sem(auprListmines1)
hauprList = std_errauprList * t.ppf((1 + confidence) / 2, nauprList  - 1)



print("mines1: "+ str(maucList) + '  -+' + str(haucList), str(mauprList) + '  -+' + str(hauprList))


# In[17]:


mypd = []
mypd = pd.DataFrame(mypd)
mypd['AUC'],mypd['AUPR'] = aucListmines1, auprListmines1
mypd.to_csv('selectedModelresult(-1).csv')


# In[ ]:




