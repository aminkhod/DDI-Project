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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Softmax, Dropout


# In[2]:


tripleData = pd.read_csv('../../triple42702.csv')
fold = 10 
interval = int(42702/fold)
auprList = []
aucList = []


# In[12]:


# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]


# In[14]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# In[15]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# #compile model using accuracy to measure model performance
# from keras import optimizers
# from keras import metrics as kmetr


# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


# In[16]:



# from keras.utils import plot_model

# plot_model(model,show_shapes = True, to_file='model_'+str(split2)+'.png')


# In[18]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# #compile model using accuracy to measure model performance
# from keras import optimizers
# from keras import metrics as kmetr


# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


# # # Load the model's saved weights.
# # model.load_weights('cnn.h5')

# #train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)


# In[23]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# #compile model using accuracy to measure model performance
# from keras import optimizers
# from keras import metrics as kmetr


# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


# # # Load the model's saved weights.
# # model.load_weights('cnn.h5')

# #train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)


# # Saveing the Model
# model.save_weights('cnn_'+str(split2)+'.h5')


# In[19]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# #compile model using accuracy to measure model performance
# from keras import optimizers
# from keras import metrics as kmetr


# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


# # # Load the model's saved weights.
# # model.load_weights('cnn.h5')

# #train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)


# # Saveing the Model
# model.save_weights('cnn_'+str(split2)+'.h5')


# #predict first 4 images in the test set
# predit = model.predict(X_test)


# In[20]:




# trainIndex = list(range(0,split1*interval))
# trainIndex.extend(list(range(split2*interval,42702)))

# testIndex = list(range(split1*interval,split2*interval))

# dataTrain = tripleData.iloc[trainIndex,:]
# dataTest = tripleData.iloc[testIndex,:]


# X_train, X_test = dataTrain.values[:,3:], dataTest.values[:,3:]
# y_train, y_test = dataTrain.values[:,2].astype(int), dataTest.values[:,2].astype(int)
# trainNum = len(X_train)
# testNum = len(X_test)

# #reshape data to fit model
# X_train = X_train.reshape(trainNum,16,71,1)
# X_test = X_test.reshape(testNum,16,71,1)

# y_train = y_train + 1
# y_test  = y_test + 1
# y_train = y_train / 2
# y_test  = y_test / 2
# # print(y_train[0], y_test[0])

# from keras.utils import to_categorical
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_test[0]
# #create model
# model = Sequential()
# #add model layers
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# # kernel_initializer='uniform',
# model.add(Conv2D(128, kernel_size=4, activation='relu', input_shape=(16,71,1)))
# # model.add(Conv2D(64, kernel_size=2, activation='relu'))

# model.add(Conv2D(32, kernel_size=4, activation='relu'))
# # model.add(Conv2D(16, kernel_size=2, activation='relu'))
# model.add(Conv2D(8, kernel_size=4, activation='relu'))
# model.add(Flatten())
# model.add(Dense( 1024, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense( 64, activation='relu'))
# model.add(Dense( 2, activation='relu'))
# model.add(Softmax(128))
# model.summary()


# #compile model using accuracy to measure model performance
# from keras import optimizers
# from keras import metrics as kmetr


# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


# # # Load the model's saved weights.
# # model.load_weights('cnn.h5')

# #train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)


# # Saveing the Model
# model.save_weights('cnn_'+str(split2)+'.h5')


# #predict first 4 images in the test set
# predit = model.predict(X_test)

# #actual results for first 4 images in test set
# print(predit[:4])


# In[7]:



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
    X_train = X_train.reshape(trainNum,16,71,1)
    X_test = X_test.reshape(testNum,16,71,1)

    y_train = y_train + 1
    y_test  = y_test + 1
    y_train = y_train / 2
    y_test  = y_test / 2
    # print(y_train[0], y_test[0])

    from keras.utils import to_categorical
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    model.add(Softmax(128))
    model.summary()


    #compile model using accuracy to measure model performance
    from keras import optimizers
    from keras import metrics as kmetr


    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # model.compile(loss='hinge', optimizer=adam, metrics=[kmetr.categorical_accuracy])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ## Minist


    # # Load the model's saved weights.
    # model.load_weights('cnn.h5')

    #train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)


    # Saveing the Model
    model.save_weights('cnn_'+str(split2)+'.h5')


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
    aucList.append(auc_val)
    auprList.append(aupr_val)
    print(aupr_val,auc_val)


# In[ ]:


print(aucList/fold,auprList/fold)


# In[24]:


# history.history['val_acc']


# In[34]:


import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(list(range(1,16)),model.history.history['acc'])
plt.plot(list(range(1,16)),model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(list(range(1,16)),model.history.history['loss'])
plt.plot(list(range(1,16)),model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[22]:


# predit
predit[:,0].shape 


# In[23]:


predicts = []
for a,b in predit:
    if a >=b:
        predicts.append(0)
    else:
        predicts.append(1)


# In[41]:



cm = confusion_matrix(list(predicts), list((dataTest.values[:,2]+1)/2))
print(cm)

CR = classification_report(list((dataTest.values[:,2]+1)/2),list(predicts))
print(CR)
# i=0
# for j in list(data.values[9500:,2]+1):
#     if j==1:
#         i +=1
# print(i)

# plt.show()
plot_confusion_matrix_from_data(list((dataTest.values[:,2]+1)/2), list(predicts))


# In[ ]:




