import time
from sklearn.model_selection import StratifiedKFold
import random
import numbers
from keras.layers import  BatchNormalization
from keras.callbacks import TensorBoard
from keras import backend as K
K.set_image_dim_ordering('tf')
import itertools
import numpy as np
import pandas as pd
np.random.seed(10)
from time import time
import numpy as np

from sklearn import metrics
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import  Dropout
from keras.layers import Flatten
from keras.layers import  MaxPooling1D, Conv1D, RepeatVector
import keras
from keras.models import Model
from keras.layers import Dense, LSTM, Input, concatenate

def Conv_LSTM(num_classes, timesteps, reg,shape):
    input_layer = Input(shape=(600,1))
    
    conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn1)
    
    conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn2)
    
    conv3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(bn3)
    gmeanpl = GlobalAveragePooling1D()(bn3)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)
    
    # fl = Flatten()(mergedlayer)
    rv = RepeatVector(300)(mergedlayer)
    lstm1 = LSTM(128,return_sequences=True)(bn3)
    do3 = Dropout(0.5)(lstm1)
    
    lstm2 = LSTM(64)(do3)
    do4 = Dropout(0.2)(lstm2)
    
    # flat = Flatten()(mergedlayer)
    output_layer = Dense(num_classes, activation='softmax')(mergedlayer)
    
    model = Model(inputs=input_layer, outputs=output_layer)  
    
    return model

def model_train(model, number_epoch, train_x, train_y):   
    optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.001)

    # a stopping function should the validation loss stop improving
    #earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    #plot_model(model, show_shapes=True, to_file='ConvLSTM.png')   
    tensorboardRNN = TensorBoard(log_dir="RNN_logs/{}".format(time()))
    
    #for i in range(number_epoch):
    history1 = model.fit(train_x, train_y, validation_split=0.1, callbacks=[tensorboardRNN], batch_size=45, epochs=int(number_epoch), shuffle=False)
    #model.reset_states()        
    
    print(model.summary())

    return model, history1

def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores

# def generatePairs(ddi_df, embedding_df):
#
#     drugs = set(ddi_df.Drug1.unique())
#     drugs = drugs.union(ddi_df.Drug2.unique())
#     drugs = drugs.intersection(embedding_df.Drug.unique())
#
#     ddiKnown = set([tuple(x) for x in  ddi_df[['Drug1','Drug2']].values])
#
#     pairs = list()
#     classes = list()
#
#     for dr1,dr2 in itertools.combinations(sorted(drugs),2):
#         if dr1 == dr2: continue
#
#         if (dr1,dr2)  in ddiKnown or  (dr2,dr1)  in ddiKnown:
#             cls=1
#         else:
#             cls=0
#
#         pairs.append((dr1,dr2))
#         classes.append(cls)
#
#     pairs = np.array(pairs)
#     classes = np.array(classes)
#
#     return pairs, classes

# def balance_data(pairs, classes, n_proportion):
#     classes = np.array(classes)
#     pairs = np.array(pairs)
#
#     indices_true = np.where(classes == 1)[0]
#     indices_false = np.where(classes == 0)[0]
#
#     np.random.shuffle(indices_false)
#     indices = indices_false[:(n_proportion*indices_true.shape[0])]
#     print ("+/-:", len(indices_true), len(indices), len(indices_false))
#     pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
#     classes = np.concatenate((classes[indices_true], classes[indices]), axis=0)
#
#     return pairs, classes

def get_scores(clf, X_new, y_new):

    scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
    scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
    #print(scorers)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores

# def crossvalid(train_df, test_df, clfs, run_index, fold_index):
#     features_cols = train_df.columns.difference(['Drug1','Drug2' ,'Class', 'Drug_x', 'Drug_y'])
#
#     X=train_df[features_cols].values
#     print(type(X))
#
#     #np.savetxt("train_x.csv", X, delimiter=",")
#
#     y=train_df['Class'].values.ravel()
#     #np.savetxt("train_y.csv", y, delimiter=",")
#
#     X_new=test_df[features_cols].values
#     #np.savetxt("test_x.csv", X_new, delimiter=",")
#
#     y_new=test_df['Class'].values.ravel()
#     #np.savetxt("test_y.csv", y_new, delimiter=",")
#
#     results = pd.DataFrame()
#     for name, clf in clfs:
#         clf.fit(X, y)
#         scores = get_scores(clf, X_new, y_new)
#         scores['method'] = name
#         scores['fold'] = fold_index
#         scores['run'] = run_index
#         results = results.append(scores, ignore_index=True)
#
#     return results, X, y, X_new, y_new

# def cv_run(run_index, pairs, classes, embedding_df, train, test, fold_index, clfs):
#     print(len(train),len(test))
#     train_df = pd.DataFrame(list(zip(pairs[train,0],pairs[train,1],classes[train])),columns=['Drug1','Drug2','Class'])
#     test_df = pd.DataFrame(list(zip(pairs[test,0],pairs[test,1],classes[test])),columns=['Drug1','Drug2','Class'])
#
#     train_df = train_df.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
#     test_df = test_df.merge(embedding_df, left_on='Drug1', right_on='Drug').merge(embedding_df, left_on='Drug2', right_on='Drug')
#
#     #train_df.to_csv('train.csv', sep=',', encoding='utf-8')
#     #test_df.to_csv('test.csv', sep=',', encoding='utf-8')
#     all_scores, train_x, train_y, test_x, test_y = crossvalid(train_df, test_df, clfs, run_index, fold_index)
#
#     np.savetxt("train_x.csv", train_x, delimiter=",")
#     np.savetxt("train_y.csv", train_y, delimiter=",")
#     np.savetxt("test_x.csv", test_x, delimiter=",")
#     np.savetxt("test_y.csv", test_y, delimiter=",")
#
#     return all_scores


# def cvSpark(sc, run_index, pairs, classes, cv, embedding_df, clfs):
#     #print (cv)
#     rdd = sc.parallelize(cv).map(lambda x: cv_run(run_index, pairs, classes, embedding_df, x[0], x[1], x[2], clfs))
#     all_scores = rdd.collect()
#     return all_scores


# def kfoldCV(sc, pairs_all, classes_all, embedding_df, clfs, n_run, n_fold, n_proportion,  n_seed):
#     scores_df = pd.DataFrame()
#     bc_embedding_df = sc.broadcast(embedding_df)
#     print(type(bc_embedding_df))
#     for r in range(n_run):
#         n_seed += r
#         random.seed(n_seed)
#         np.random.seed(n_seed)
#         n_proportion = 1
#         pairs, classes= balance_data(pairs_all, classes_all, n_proportion)
#
#         skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=n_seed)
#         cv = skf.split(pairs, classes)
#
#         print ('run',r)
#         bc_pairs_classes = sc.broadcast((pairs, classes))
#         cv_list = [ (train,test,k) for k, (train, test) in enumerate(cv)]
#         #pair_df = pd.DataFrame(list(zip(pairs_[:,0],pairs_[:,1],classes_)), columns=['Drug1','Drug2','Class'])
#         scores = cvSpark(sc, r, bc_pairs_classes.value[0], bc_pairs_classes.value[1], cv_list, bc_embedding_df.value, clfs)
#         scores_df = scores_df.append(scores)
#
#     return scores_df
