# 2020.06.04
import numpy as np
import sklearn
import sys
from sklearn.svm import SVC
import time

from framework.pixelhop2 import PixelHopPP_Unit
from framework.utli import read_dataset, lfw_train_test, myStandardScaler, Generate_feature, MaxPooling


root = '/mnt/yifan/face/'
pair_txt_path ='data/pairs.txt'
images_path = "data/HEFrontalizedLfw2/"

# number of kernels retained in each hop
n1 =18
n2 = 13
n3 = 11

standardize = True
energy_th = 0.0005
num_of_train_pixelhop = 4000

foldnum = 1
if len(sys.argv) > 2:
    foldnum = int(sys.argv[1])
    print("\n   <Setting> 10-fold validation. Use: fold%s"%(foldnum))
else:
    print("\n   <Warning> 10-fold validation. No fold number provided, use: fold%s as default!"%(foldnum))

print("   <Setting> root location: %s"%(root))
print("   <Setting> relative image location: %s"%(images_path))
print("   <Setting> relative pairs.txt location: %s\n"%(pair_txt_path))
print("   <Setting> %s nodes are retained Hop1's Saab transfroation"%(str(n1)))
print("   <Setting> %s nodes are retained Hop2's Saab transfroation"%(str(n2)))
print("   <Setting> %s nodes are retained Hop3's Saab transfroation"%(str(n3)))
print("   <Setting> Energy thresold for discarding nodes (TH2): %s\n"%(str(energy_th)))

def main(argv):
    t0 = time.time()
    #determine the size of the input as size
    raw_images, flipped_raw_images, raw_labels = read_dataset(path=root+images_path, size=32)
    trainData1, trainData2, trainLabel, testData1, testData2, testLabel = lfw_train_test(root, pair_txt_path, raw_images, flipped_raw_images, raw_labels, foldnum, includeflipped=True)

    print("\n     <INFO> Use %s images to train Saab kernels!"%(num_of_train_pixelhop))
    S = [len(trainData1), len(trainData2), len(testData1), len(testData2)]
    allpatches = np.concatenate((trainData1,trainData2,testData1,testData2),axis=0)
    assert (num_of_train_pixelhop < len(trainData1)+len(trainData2)), " Too many images to train Saab Kernels!"
    #Train first layer of saab

    print("   <Time> Load images and get pair: %s seconds"%(str(time.time()-t0)))
    t1 = time.time()

    saab, outtrainsaab1, kernel_filter, falttened = PixelHopPP_Unit(allpatches[0:num_of_train_pixelhop], num_kernels=n1, saab=None, window=5, stride=1, train=True, energy_th=energy_th, ch_decoupling=False, ch_energies=None, kernel_filter=[])
    #Apply saab transfrom to the training data
    _, out1, _, _ = PixelHopPP_Unit(allpatches, num_kernels=n1, saab=saab, window=5, stride=1, train=False, ch_decoupling=False, kernel_filter=kernel_filter)
    out1ave = MaxPooling(out1)
    print("       <INFO> Hop1 #Nodes: %s"%(out1.shape[-1]))
    saab, outtrainsaab2, kernel_filter, falttened = PixelHopPP_Unit(out1ave[0:num_of_train_pixelhop], num_kernels=n2, saab=None, window=5, stride=1, train=True, energy_th=energy_th, ch_decoupling=True, ch_energies=falttened, kernel_filter=[])
    _, out2, _, _ = PixelHopPP_Unit(out1ave, num_kernels=n2, saab=saab, window=5, stride=1, train=False, ch_decoupling=True, kernel_filter=kernel_filter)
    out2ave = MaxPooling(out2)
    print("       <INFO> Hop2 #Nodes: %s"%(out2.shape[-1]))
    saab, _, kernel_filter, falttened = PixelHopPP_Unit(out2ave[0:num_of_train_pixelhop], num_kernels=n3, saab=None, window=5, stride=1, train=True, energy_th=energy_th, ch_decoupling=True, ch_energies=falttened, kernel_filter=[])
    _, out3, _, _ = PixelHopPP_Unit(out2ave,num_kernels=n3,saab=saab,window=5,stride=1,train=False,ch_decoupling=True,kernel_filter=kernel_filter)
    print("       <INFO> Hop2 #Nodes: %s"%(out3.shape[-1]))

    out1_1, out1_2, out1test, out1test_2 = out1[0:S[0]], out1[S[0]:S[1]+S[0]], out1[S[1]+S[0]:S[2]+S[1]+S[0]], out1[S[2]+S[1]+S[0]:S[3]+S[2]+S[1]+S[0]]
    out2_1, out2_2, out2test, out2test_2 = out2[0:S[0]], out2[S[0]:S[1]+S[0]], out2[S[1]+S[0]:S[2]+S[1]+S[0]], out2[S[2]+S[1]+S[0]:S[3]+S[2]+S[1]+S[0]]   
    out3_1, out3_2, out3test, out3test_2 = out3[0:S[0]], out3[S[0]:S[1]+S[0]], out3[S[1]+S[0]:S[2]+S[1]+S[0]], out3[S[2]+S[1]+S[0]:S[3]+S[2]+S[1]+S[0]]

    print("   <Time> c/w Saab: %s seconds"%(str(time.time()-t1)))
    t1 = time.time()

    if standardize == True:
        print("       <INFO> Standard each frequency components to mean 0 std 1...")
        _, S = myStandardScaler(np.concatenate((out1_1, out1_2), axis=0), [], True)
        out1_1, _ = myStandardScaler(out1_1, S, False)
        out1_2, _ = myStandardScaler(out1_2, S, False)
        out1test, _ = myStandardScaler(out1test, S, False)
        out1test_2, _ = myStandardScaler(out1test_2, S, False)

        _, S = myStandardScaler(np.concatenate((out2_1, out2_2), axis=0), [], True)
        out2_1, _ = myStandardScaler(out2_1, S, False)
        out2_2, _ = myStandardScaler(out2_2, S, False)
        out2test , _ = myStandardScaler(out2test, S, False)
        out2test_2 , _= myStandardScaler(out2test_2, S, False)

        _, S = myStandardScaler(np.concatenate((out3_1, out3_2), axis=0), [], True)
        out3_1, _ = myStandardScaler(out3_1, S, False)
        out3_2, _ = myStandardScaler(out3_2, S, False)
        out3test, _ = myStandardScaler(out3test, S, False)
        out3test_2, _ = myStandardScaler(out3test_2, S, False)
        print("   <Time> Standardize feature: %s seconds"%(str(time.time()-t1)))
    t1 = time.time()

    x1 = Generate_feature(out1_1, out1_2, 1)
    x1t = Generate_feature(out1test, out1test_2, 1)
    print("       <INFO> Hop1 get %s attributes!"%(x1.shape[-1]))
    x2 = Generate_feature(out2_1, out2_2, 2)
    x2t = Generate_feature(out2test, out2test_2, 2)
    print("       <INFO> Hop2 get %s attributes!"%(x1.shape[-1]))
    x3 = Generate_feature(out3_1, out3_2, 3)
    x3t = Generate_feature(out3test, out3test_2, 3)
    print("       <INFO> Hop3 get %s attributes!"%(x1.shape[-1]))
    finalTrainData = np.concatenate((x1,x2,x3),axis=-1)
    finalTrainLabel = np.asarray(trainLabel)
    finalTestData = np.concatenate((x1t,x2t,x3t),axis=-1)
    finalTestLabel = np.asarray(testLabel)  

    print("\n     <INFO> Final train data shape %s"%(str(finalTrainData.shape)))
    print("     <INFO> Final test data shape %s"%(str(finalTestData.shape)))
    print("   <Time> Generate feature: %s seconds"%(str(time.time()-t1)))
    t1 = time.time()

    print("\n   <INFO> Using SVM to learn the features.")
    clf = SVC(gamma='auto', probability=True)
    clf.fit(finalTrainData,finalTrainLabel)
    print("     <INFO> train acc: %s"%(clf.score(finalTrainData,finalTrainLabel)))
    print("     <INFO> test acc: %s"%(clf.score(finalTestData,finalTestLabel)))

    print("   <Time> SVM: %s seconds"%(str(time.time()-t1)))
    print(" <Time> Total usage: %s seconds"%(str(time.time()-t0)))

main(sys.argv[1:])