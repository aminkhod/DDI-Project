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
from keras import optimizers
from keras import metrics as kmetr
from keras.utils import plot_model


# In[2]:


tripleData = pd.read_csv('../../triple42702_ShuffledPaired.csv')
zerodatad = pd.read_csv('../../probZeroDrug30000.csv')


# In[3]:


fold = 10 
interval = int(42702/fold)
zeroInterval = int(30000/fold)

auprListZero = []
aucListZero = []

aucListpositive1 = []
auprListpositive1 = []

aucListmines1 = []
auprListmines1 = []


# In[5]:


def build_model():
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
    model.add(Dropout(0.2))
    model.add(Dense( 16, activation='relu'))
    model.add(Dense( 3, activation='sigmoid'))
    # model.add(Softmax(128))
    model.summary()

    #compile model using accuracy to measure model performance


    adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# In[6]:


#from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from keras.utils import to_categorical


for split1 in range(0,10):
    split2 = split1 + 1
    
    trainIndex = list(range(0,split1 * interval))
    trainIndex.extend(list(range(split2 * interval,42702)))
    zeroTrainIndex = list(range(0,split1 * zeroInterval))
    zeroTrainIndex.extend(list(range(split2 * zeroInterval,30000)))

    if split1==9:
        testIndex = list(range(split1 * interval,42702))
        zeroTestIndex = list(range(split1 * zeroInterval,30000))

    else: 
        testIndex = list(range(split1 * interval,split2 * interval))
        ZeroTestIndex = list(range(split1 * zeroInterval,split2 * zeroInterval))
    
    dataTrain = pd.concat([tripleData.iloc[trainIndex,:], zerodatad.iloc[zeroTrainIndex,:]], ignore_index = True)
    dataTest = pd.concat([tripleData.iloc[testIndex,:], zerodatad.iloc[ZeroTestIndex,:]], ignore_index = True)

#     print(dataTrain.shape)
    print(min(testIndex),max(testIndex))
#     print(len(dataTest.iloc[:,0]), len(dataTrain.iloc[:,0]))


    X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
    y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2]
    trainNum = len(X_train)
    testNum = len(X_test)

    #reshape data to fit model
    X_train = X_train.reshape(trainNum,16,71,1).astype('float32')
    X_test = X_test.reshape(testNum,16,71,1).astype('float32')

    y_train = y_train + 1
    y_test  = y_test + 1

    # print(y_train[0], y_test[0])


    #one-hot encode target column
    y_train = to_categorical(y_train).astype(int)
    y_test = to_categorical(y_test).astype(int)
    print(y_train[0],y_test[0])
    
    #create model
    model = build_model()
    # # Load the model's saved weights.
    # model.load_weights('cnn.h5')

    #train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=9)


    # Saveing the Model
    model.save_weights('Weight/10-foldCV_LastTripleDDI/cnnTriple_'+str(split2)+'.h5')


    #predict first 4 images in the test set
    predit = model.predict(X_test)

    #actual results for first 4 images in test set
    print(predit[:4])



    # Degressive Result
    prec, rec, thr = precision_recall_curve(y_test[:,0], predit[:,0])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_test[:,0], predit[:,0])
    auc_val = auc(fpr, tpr)
    aucListmines1.append(auc_val)
    auprListmines1.append(aupr_val)
    print(aupr_val,auc_val)

    # Enhancive Result
    prec, rec, thr = precision_recall_curve(y_test[:,2], predit[:,2])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_test[:,2], predit[:,2])
    auc_val = auc(fpr, tpr)
    aucListpositive1.append(auc_val)
    auprListpositive1.append(aupr_val)
    print(aupr_val , auc_val )
    
    # Zero Result
    prec, rec, thr = precision_recall_curve(y_test[:,1], predit[:,1])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(y_test[:,1], predit[:,1])
    auc_val = auc(fpr, tpr)
    aucListZero.append(auc_val)
    auprListZero.append(aupr_val)
    print(aupr_val , auc_val )    


# In[ ]:





# In[6]:


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


# In[7]:


mypd = []
mypd = pd.DataFrame(mypd)
mypd['AUC'],mypd['AUPR'] = aucListpositive1, auprListpositive1
mypd.to_csv('LastTripleModelresult(+1).csv')


# In[8]:


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


# In[9]:


mypd = []
mypd = pd.DataFrame(mypd)
mypd['AUC'],mypd['AUPR'] = aucListmines1, auprListmines1
mypd.to_csv('LastTripleModelresult(-1).csv')


# In[10]:


confidence = 0.95
# data = [1, 2, 3, 4, 5]

naucList = len(aucListZero)
maucList = mean(aucListZero)
std_erraucList = sem(aucListZero)
haucList = std_erraucList * t.ppf((1 + confidence) / 2, naucList - 1)


nauprList = len(auprListZero)
mauprList = mean(auprListZero)
std_errauprList = sem(auprListZero)
hauprList = std_errauprList * t.ppf((1 + confidence) / 2, nauprList  - 1)


print("Zeros: "+ str(maucList) + '  -+' + str(haucList), str(mauprList) + '  -+' + str(hauprList))


# In[11]:


mypd = []
mypd = pd.DataFrame(mypd)
mypd['AUC'],mypd['AUPR'] = aucListZero, auprListZero
mypd.to_csv('LastTripleModelresult(Zero).csv')


# In[ ]:




