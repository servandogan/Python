import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random
from sklearn.model_selection import KFold
df1 = pd.read_csv("train1.csv") 
df2 = pd.read_csv("train2.csv") 
df3 = pd.read_csv("train3.csv")


df1_x1 = df1['x1']
df1_x2 = df1['x2']
df1_numpy = df1.values
df1_x1_numpy = df1['x1'].values

import statistics

df1_class0 = df1[df1['y']==0]
df1_class1 = df1[df1['y']==1]
df1_class0_x1 = df1_class0['x1']
df1_class0_x2 = df1_class0['x2']
df1_class1_x1 = df1_class1['x1']
df1_class1_x2 = df1_class1['x2']

df2_class0 = df2[df2['y']==0]
df2_class1 = df2[df2['y']==1]
df2_class0_x1 = df2_class0['x1']
df2_class0_x2 = df2_class0['x2']
df2_class1_x1 = df2_class1['x1']
df2_class1_x2 = df2_class1['x2']

figf1class0 = plt.figure()
figf1class1 = plt.figure()
figf2class0 = plt.figure()
figf2class1 = plt.figure()


axf1class0 = figf1class0.add_subplot(111)
axf1class1 = figf1class1.add_subplot(111)
axf2class0 = figf2class0.add_subplot(111)
axf2class1 = figf2class1.add_subplot(111)



df1_class0_x1_numpy = df1_class0_x1.values
df1_class0_x2_numpy = df1_class0_x2.values
df1_class1_x1_numpy = df1_class1_x1.values
df1_class1_x2_numpy = df1_class1_x2.values

df2_class0_x1_numpy = df2_class0_x1.values
df2_class0_x2_numpy = df2_class0_x2.values
df2_class1_x1_numpy = df2_class1_x1.values
df2_class1_x2_numpy = df2_class1_x2.values

cov_mtrxfeat1class0 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        cov_mtrxfeat1class0[i][j] = np.cov(df1_class0_x1_numpy, df1_class0_x2_numpy, bias=True)[i][j]
print(statistics.mean(df1_class0_x1_numpy))
print(statistics.mean(df1_class0_x2_numpy))
print(cov_mtrxfeat1class0)

cov_mtrxfeat1class1 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        cov_mtrxfeat1class1[i][j] = np.cov(df1_class1_x1_numpy, df1_class1_x2_numpy, bias=True)[i][j]
print(statistics.mean(df1_class1_x1_numpy))
print(statistics.mean(df1_class1_x2_numpy))
print(cov_mtrxfeat1class1)



cov_mtrxfeat2class0 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        cov_mtrxfeat2class0[i][j] = np.cov(df2_class0_x1_numpy, df2_class0_x2_numpy, bias=True)[i][j]
print(cov_mtrxfeat2class0)

cov_mtrxfeat2class1 = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        cov_mtrxfeat2class1[i][j] = np.cov(df2_class1_x1_numpy, df2_class1_x2_numpy, bias=True)[i][j]
print(cov_mtrxfeat2class1)





alpha = ['x1', 'x2']
caxf1class0 = axf1class0.matshow(cov_mtrxfeat1class0,interpolation='nearest') #cov. matrix of dataset1
figf1class0.colorbar(caxf1class0)
axf1class0.set_xticklabels(['']+alpha)
axf1class0.set_yticklabels(['']+alpha)


caxf1class1 = axf1class1.matshow(cov_mtrxfeat1class1,interpolation='nearest') #cov. matrix of dataset1
figf1class1.colorbar(caxf1class1)
axf1class1.set_xticklabels(['']+alpha)
axf1class1.set_yticklabels(['']+alpha)


caxf2class0 = axf2class0.matshow(cov_mtrxfeat2class0,interpolation='nearest') #cov. matrix of dataset1
figf2class0.colorbar(caxf2class0)
axf2class0.set_xticklabels(['']+alpha)
axf2class0.set_yticklabels(['']+alpha)

caxf2class1 = axf2class1.matshow(cov_mtrxfeat2class1,interpolation='nearest') #cov. matrix of dataset1
figf2class1.colorbar(caxf2class1)
axf2class1.set_xticklabels(['']+alpha)
axf2class1.set_yticklabels(['']+alpha)



plt.figure()
plt.gca().set(title='Data distribution of feature 1 of trainingSet 1', ylabel='Frequency')
plt.hist(df1_class0_x1_numpy, alpha=0.5)
plt.hist(df1_class1_x1_numpy, alpha=0.5)
plt.figure()
plt.gca().set(title='Data distribution of feature 2 of trainingSet 1', ylabel='Frequency')
plt.hist(df1_class0_x2_numpy, alpha=0.5)
plt.hist(df1_class1_x2_numpy, alpha=0.5)
plt.figure()
plt.gca().set(title='Data distribution of feature 1 of trainingSet 2', ylabel='Frequency')
plt.hist(df2_class0_x1_numpy, alpha=0.5)
plt.hist(df2_class1_x1_numpy, alpha=0.5)
plt.figure()
plt.gca().set(title='Data distribution of feature 2 of trainingSet 2', ylabel='Frequency')
plt.hist(df2_class0_x2_numpy, alpha=0.5)
plt.hist(df2_class1_x2_numpy, alpha=0.5)



df1_x1 = df1['x1']
df1_x2 = df1['x2']
df2_x1 = df2['x1']
df2_x2 = df2['x2']

df1_x1_numpy = df1_x1
df1_x2_numpy = df1_x2
df2_x1_numpy = df2_x1
df2_x2_numpy = df2_x2



scatter_f1_class0 = plt.figure()
plt.title("Scatterplot for training dataset 1")
plt.scatter(df1_x1_numpy, df1_x2_numpy)
plt.scatter(df1_x2_numpy, df1_x1_numpy)

scatter_f2_class0 = plt.figure()
plt.title("Scatterplot for training dataset 2")
plt.scatter(df2_x1_numpy, df2_x2_numpy)
plt.scatter(df2_x2_numpy, df2_x1_numpy)

a = np.array([cov_mtrxfeat2class0,cov_mtrxfeat2class1])

test1 = pd.read_csv("test1.csv")
test2 = pd.read_csv("test2.csv")


def calculatecovMtrx(df_x1_numpy, df_x2_numpy):
    cov_mtrx= np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            cov_mtrx[i][j] = np.cov(df_x1_numpy, df_x2_numpy, bias=True)[i][j]
    return cov_mtrx

def getClassesFromtrainSamples(dfx):
    if(type(dfx) != type(np.zeros((1,1)))):
        df_x1 = dfx['x1']
        df_x2 = dfx['x2']
        df_clss0 = dfx[dfx['y']==0]
        df_clss1 = dfx[dfx['y']==1]
        df_clss0_x1 = (df_clss0['x1']).values
        df_clss0_x2 = (df_clss0['x2']).values
        df_clss1_x1 = (df_clss1['x1']).values
        df_clss1_x2 = (df_clss1['x2']).values
    else:
        index = []
        for i in range(len(dfx)):
            index.append(i)
        df_x1 = dfx[0:len(dfx),0]
        df_x1 = pd.DataFrame(df_x1, index)
        df_x2 = dfx[0:len(dfx),1]
        df_x2 = pd.DataFrame(df_x2, index)
        df_clss0cool = dfx[dfx[0:len(dfx),2]==0]
        index = []
        for i in range(len(df_clss0cool)):
            index.append(i)
        df_clss0 = pd.DataFrame(df_clss0cool, index)
        df_clss1cool = dfx[dfx[0:len(dfx),2]==1]
        index = []
        for i in range(len(df_clss1cool)):
            index.append(i)
        df_clss1 = pd.DataFrame(df_clss1cool, index)
        df_clss0_x1 = df_clss0cool[0:len(dfx),0]
        df_clss0_x2 = df_clss0cool[0:len(dfx),1]
        df_clss1_x1 = df_clss1cool[0:len(dfx),0]
        df_clss1_x2 = df_clss1cool[0:len(dfx),1]

    
    return df_x1, df_x2, df_clss0, df_clss1, df_clss0_x1, df_clss0_x2, df_clss1_x1, df_clss1_x2


def trainBayes(trainSamples, whichClassUWannaKnow, x):

    df_x1, df_x2, clss0, clss1, clss0_x1, clss0_x2, clss1_x1, clss1_x2 = getClassesFromtrainSamples(trainSamples)

    if(whichClassUWannaKnow == 0):
        meanDS2 = np.zeros((2,1))
        meanDS2[0][0] = statistics.mean(clss0_x1)
        meanDS2[1][0] = statistics.mean(clss0_x2)
        covMtrx1 = calculatecovMtrx(clss0_x1, clss0_x2)
        probability = len(clss0)/len(clss1 + clss0)
    else:
        meanDS2 = np.zeros((2,1))
        meanDS2[0][0] = statistics.mean(clss1_x1)
        meanDS2[1][0] = statistics.mean(clss1_x2)
        covMtrx1 = calculatecovMtrx(clss1_x1, clss1_x2)
        probability = len(clss1)/len(clss1 + clss0)
    cov_mtrxtogether = calculatecovMtrx(df_x1, df_x2) # SIGMA_i

    comparecovmtrx1 = calculatecovMtrx(clss0_x1, clss0_x2)
    comparecovmtrx2 = calculatecovMtrx(clss1_x1, clss1_x2)

    if(np.allclose(comparecovmtrx1,comparecovmtrx2)):
        g_i = np.dot(np.dot(np.matrix.transpose(meanDS2),np.linalg.inv(cov_mtrxtogether)),x) + (np.log(probability)-1/2 * np.dot(np.dot(np.matrix.transpose(meanDS2),np.linalg.inv(cov_mtrxtogether)),meanDS2))
        return g_i
    else:
        g_i = np.dot(np.dot(np.matrix.transpose(x),(-1/2*(np.linalg.inv(covMtrx1)))),x) + np.dot(np.dot(np.matrix.transpose(meanDS2), np.linalg.inv(covMtrx1)), x)+ (-1/2 * np.dot(np.dot((np.matrix.transpose(meanDS2)),np.linalg.inv(covMtrx1)),meanDS2) +( - 1/2* np.log(np.linalg.det(covMtrx1)))+ np.log(probability)) 
        return g_i

    
def abc(trainSamples, testSamples, bothclass=0):
    if(type(testSamples) != type(np.zeros((1,1)))):
        testSamples = testSamples.values
    z = np.empty((1,1))
    w = np.empty((1,1))
    icountclss1 = []
    icountclss2 = []
    for i in range(len(testSamples)):
        x = np.array([testSamples[i][0],testSamples[i][1]])
        y = np.zeros((2,1))
        y[0][0] = x[0]
        y[1][0] = x[1]
        x = y
        if(bothclass==0):
            a = trainBayes(trainSamples, 0, x)
            b = trainBayes(trainSamples, 1, x)
            if(np.less(b,a)):
                #class1
                icountclss1.append(0)
                icountclss2.append("n")
                z = np.append(z, a, axis=0)
            elif(np.less(a,b)):
                #class2
                icountclss2.append(1)
                icountclss1.append("n")
                w = np.append(w, b, axis=0)

        
        elif(bothclass==1):
            a = trainBayes(trainSamples, 0, x)
            #class1
            if(a == 0):
                icountclss1.append(0)
            else:    
                icountclss1.append("n")
            z = np.append(z, a, axis=0)
        else:
            #class2
            a = trainBayes(trainSamples, 1, x)
            if(a == 1):
                icountclss1.append(1)
            else:    
                icountclss1.append("n")
            z = np.append(z, a, axis=0)
    if(bothclass == 0):
        z = np.delete(z,0)
        w = np.delete(w,0)
        return z, w, icountclss1, icountclss2
    
    else:
        z = np.delete(z,0)
        return z, icountclss1
        



def mismatch(df1, test1, df2, test2, bothclass=0):
    
    if(bothclass==0):
        database1clss1, database1clss2, db1icountclss1, db1icountclss2 = abc(df1, test1)
        database2clss1, database1clss2, db2icountclss1, db2icountclss2 = abc(df2, test2)
    else:
        database1clss1, db1icountclss1 = abc(df1, test1, 1)
        database2clss2, db2icountclss2 = abc(df2, test2, 2)
    if(type(test1) != type(np.zeros((1,1)))):
        test1 = test1.values
        test2 = test2.values
    
    
    properReihenfolge1 = []
    properReihenfolge2 = []

    for i in range(len(test1)):
        properReihenfolge1.append(test1[i][2])

    for i in range(len(test2)):
        properReihenfolge2.append(test2[i][2])
    correct = 0
    correct2 = 0
    mismatch = len(test1)
    mismatch2 = len(test2)
    if(bothclass==0):
        for i in range(len(test1)):
            if(properReihenfolge1[i] == db1icountclss1[i] or (properReihenfolge1[i]==1 and db1icountclss1[i]=='n')):
                correct += 1

        for i in range(len(test2)):
            if(properReihenfolge2[i] == db2icountclss1[i] or (properReihenfolge2[i]==1 and db2icountclss1[i]=='n')):
                correct2 += 1
        mismatch = mismatch - correct
        mismatch2 = mismatch2 - correct2

        e_1 = mismatch/len(test1) * 100
        e_2 = mismatch2/len(test2) * 100
        
    else:
        for i in range(len(test1)):
            if(properReihenfolge1[i] == db1icountclss1[i] or (properReihenfolge1[i]==1 and db1icountclss1[i]=='n')):
                correct += 1

        for i in range(len(test2)):
            if(properReihenfolge2[i] == db2icountclss2[i] or (properReihenfolge2[i]==1 and db2icountclss2[i]=='n')):
                correct2 += 1
        mismatch = mismatch - correct
        mismatch2 = mismatch2 - correct2

        e_1 = mismatch/len(test1) * 100
        e_2 = mismatch2/len(test2) * 100
    
    return e_1, e_2
mismatch(df1,test1,df2,test2)



trainPLUStest1 = (pd.concat([df1,test1])).values
trainPLUStest2 = (pd.concat([df2,test2])).values


kf = KFold(n_splits=5, shuffle=True, random_state=random.seed(1773))
kf.get_n_splits(trainPLUStest1)


errors = np.zeros((5,2))
error1 = 0
error2 = 0
i=0
for train_index, test_index in kf.split(trainPLUStest1):
    trainPLUStest1_train, trainPLUStest1_test = trainPLUStest1[train_index], trainPLUStest1[test_index]
    trainPLUStest2_train, trainPLUStest2_test = trainPLUStest2[train_index], trainPLUStest2[test_index]
    errors[i][0],errors[i][1]= mismatch(trainPLUStest1_train, trainPLUStest1_test, trainPLUStest2_train, trainPLUStest2_test)
    error1 += errors[i][0]
    error2 += errors[i][1]
    i += 1
error1 = error1/5
error2 = error2/5


for i in range(len(errors)):
    print("err_1","err_2", i+1)
    print(errors[i][0],errors[i][1])
    
print()
print("avg_err1: ",error1)
print("avg_err2: ",error2)


test3 = pd.read_csv("test3.csv")
scatter_f3_class0 = plt.figure()
df3_x1 = df3['x1']
df3_x2 = df3['x2']
df3_clss0 = df3[df3['y']==0]
df3_clss1 = df3[df3['y']==1]
df3_x1_numpy = df3_x1
df3_x2_numpy = df3_x2

plt.scatter(df3_x1_numpy, df3_x2_numpy)
plt.scatter(df3_x2_numpy, df3_x1_numpy)


a, b = mismatch(df3, test3, df3, test3, bothclass=0)
print(a) 
c, d = mismatch(df3_clss0, test3, df3_clss1, test3, bothclass=1)
print("class0 mismatch: ",c)
print("class1 mismatch: ",d)


# The Bayes classifier is primarily good for testing whether the features have been chosen correctly.
# In this way, new test sets with unknown classes can get the correct classification with the condition 
# that the training data sets are large enough.
# With smaller data sets, however, the Bayes classifier does not work reliably regardless of whether 
# the data is shuffled or kFold is applied to it.
# kFold turns out to be ideal for larger data sets for minimizing the error rate. 
# However, care must be taken that the training data does not become too small due to kFolding.
# The Bayes Classifier is a reliable method for classifying the test data for training data sets of a 
# larger order and correctly selected classes.
# For smaller data sets should other methods be considered