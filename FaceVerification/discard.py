# 2020.06.05
# discard some data by cutting off max and min mean of cosine features
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC

def load(name):
    with open(name, 'rb') as f:
        d = pickle.load(f)
    return d['x'], d['y'], d['xt'], d['yt']

def get_cos(x, y, xt, yt):
    print('   raw', x.shape, y.shape, xt.shape, yt.shape)
    xx, xxt = [], []
    for i in range(x.shape[0]):
        xx.append(x[i][x[i]<1])
    for i in range(xt.shape[0]):
        xxt.append(xt[i][xt[i]<1])
    return np.array(xx), y, np.array(xxt), yt

def SVM(x, y, xt, yt):
    clf = SVC(gamma='auto', probability=True)
    clf.fit(x,y)
    a = clf.score(x,y)
    b = clf.score(xt,yt)
    print('       train acc ', a, '       test acc ',b)
    return a, b

def discard(x, y, xt, yt, percent=0.1):
    mx = np.mean(x, axis=1)
    idx = np.argsort(mx)
    s = (int)(idx.shape[0]*(percent))//2
    idx = idx[s:-s]
    return idx

per = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
ax = np.zeros((10, len(per)))
axt = np.zeros((10, len(per)))
for i in range(1,11):
    for j in range(len(per)):
        print("   <INFO> Run on fold%s, discard %s percent samples"%(str(i), str(per[j]*100)))
        x, y, xt, yt = load(str(i)+'.pkl')
        x1, y1, xt1, yt1 = get_cos(x, y, xt, yt)
        idx = discard(x1, y1, xt1, yt1, per[j])
        print('       shape', x[idx].shape, y[idx].shape, xt.shape, yt.shape)
        #with open(str(i)+'_discard.pkl','wb') as f:
        #    pickle.dump({'x':x[idx], 'y':y[idx],
        #                    'xt':xt, 'yt':yt}, f, protocol=4)
        ax[i-1,j], axt[i-1,j] = SVM(x[idx], y[idx], xt, yt)

x = np.mean(axt,axis=0)
xx = np.mean(ax,axis=0)
N =(1-np.array(per))*100
plt.figure(figsize=(9,6))
plt.plot(N, xx,label='train', color='blue')
plt.plot(N, x, label='test', color='red')
plt.fill_between(N, x+np.std(axt,axis=0), x-np.std(axt,axis=0), facecolor='red', alpha=0.1)
plt.fill_between(N, xx+np.std(ax,axis=0), xx-np.std(ax,axis=0), facecolor='blue', alpha=0.1)
plt.xlabel("Percent of pairs retained")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('figure1.png')
plt.show()

