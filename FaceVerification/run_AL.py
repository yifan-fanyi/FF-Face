# 2020.06.05
import numpy as np 
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

from framework.activate_learning import entropy_query, coreset, QBC

path = '../'
foldnum = (int)(sys.argv[1])
mode = 'coreset'
n_iter = 100
n_queres_per_itt = 100

with open(path+str(foldnum)+'_discard.pkl', 'rb') as f:
    d = pickle.load(f)
x_pool, y_pool, x_test, y_test = d['x'],d['y'],d['xt'],d['yt']

if mode == 'qbc':
    clf = QBC(init=0.05, n_increment=n_queres_per_itt, n_iter=n_iter,
              learners=[SVC(gamma='auto', probability=True)])
    labeled_train_data_history, train_acc_history, test_acc_history = clf.fit(x_pool, y_pool, x_test, y_test)

else:
    x_train, x_pool, y_train, y_pool = train_test_split(x_pool, y_pool, train_size=0.01, random_state=42, stratify=y_pool)
    labeled_train_data_history, train_acc_history, test_acc_history = [], [], []

    for i in range(n_iter):
        if x_pool.shape[0] < n_queres_per_itt:
            break
        print('iter:',i ,'  pool shape:', x_pool.shape, '  train shape:', x_train.shape)
        clf = SVC(gamma='auto', probability=True)
        clf.fit(x_train,y_train)
        prediction = clf.predict(x_train)
        train_acc_history.append(accuracy_score(y_train, prediction))
        prediction = clf.predict(x_test)
        test_acc_history.append(accuracy_score(y_test, prediction))
        labeled_train_data_history.append(x_train.shape[0])
        print('   train accuracy',train_acc_history[i])
        print('   test accuracy',test_acc_history[i])
        pool_prob=clf.predict_proba(x_pool)
        if mode == 'entropy':
            train_idx = entropy_query(pool_prob,n_queres_per_itt)
        elif mode == 'coreset':
            train_idx = coreset(x_pool, x_train, n_queres_per_itt)
        x_train = np.concatenate((x_train,x_pool[train_idx,:]),axis=0)
        y_train = np.concatenate((y_train,y_pool[train_idx]),axis=0)
        x_pool = x_pool[(1-train_idx).astype(bool),:]
        y_pool = y_pool[(1-train_idx).astype(bool)]

with open(mode+'_0327'+str(foldnum)+'.pkl', 'wb') as f:
    pickle.dump({'shape':labeled_train_data_history, 'train':train_acc_history, 'test':test_acc_history}, f)