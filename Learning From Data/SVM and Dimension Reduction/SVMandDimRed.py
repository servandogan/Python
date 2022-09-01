import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(csvname):
    # load in data
    data = np.asarray(pd.read_csv(csvname,header = None))

    X = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    x_ = X
    X = np.concatenate((o,X),axis = 1)
    return X,y,x_



def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def gradient_descent_soft_cost(X,y,w,alpha,lam,x_):
    # start gradient descent loop
    max_its = 10000
    w_ = np.zeros((2,1))
    
    b = w[0]
    w_[0] = w[1]
    w_[1] = w[2]
    grad_w = 0
    for k in range(max_its):
        #print('X.T: ', X.T.shape)
        #print('w: ', w.shape)
        #print('y:', y.shape)
        #print('b+X@w', (y*(b+X@w)).shape)
        #grad_w = 2*w_ - (x_+1)*y*np.exp(-y*(x_*w_+w[0]))/ (np.exp(-y*(x_*w_+w[0])) + 1)
        s = sigmoid(-y*(b+X@w))
        #print('s', s.shape)
        #print('s*y', (s*y).shape)

        grad_w = -X.T @ (s * y) + lam*2*np.sum(w_)
        
        #s = sigmoid(-y * (X @ w))
        #grad_w = -X.T @ (s * y)
        #grad_w = np.sum((s*y) * X.T, 0, keepdims=True)
        #print(grad_w)
        
        w = w - alpha * grad_w
    
    print(w)
    return w


def plot_all(X,y,w,lam):
    # custom colors for plotting points
    red = [1,0,0.4]  
    blue = [0,0.4,1]
    
    # scatter plot points
    fig = plt.figure(figsize = (4,4))
    ind = np.argwhere(y==1)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1],X[ind,2],color = red,edgecolor = 'k',s = 25)
    ind = np.argwhere(y==-1)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1],X[ind,2],color = blue,edgecolor = 'k',s = 25)
    plt.grid('off')
    
    # plot separator
    s = np.linspace(-1,1,100) 

    plt.plot(s,(-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 2)
    
    
    plt.plot(s,(1-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 1, linestyle = '--')
    plt.plot(s,(-1-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 1, linestyle= '--')

    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.title('soft-margin svm with lambda = ' + str(lam))
    plt.show()



X,y,x_ = load_data('overlapping_2class.csv')

alpha = 10**(-2)
w0 = np.random.randn(3,1)

lams = [0, 10**-2, 10**-1, 1, 10]
for lam in lams:
    # run gradient descent
    w = gradient_descent_soft_cost(X,y,w0,alpha,lam, x_)

    # plot points and separator
    plot_all(X,y,w,lam)



def sigmoid(x):
    return 1/ (1 + np.exp(-x))
    
def softmax_grad(X,y):
    # Initializations 
    # X.shape == (40,3)
    # 
    #print(X)
    #print(X.shape)
    w = np.random.randn(3,1);        # random initial point
    alpha = 10**-2
    max_its = 2000
    
    for k in range(max_its):
        
        ## fill here 

        s = sigmoid(-y * (X @ w))
        grad_w = -X.T @ (s * y)
        #grad_w = np.sum((s*y) * X.T, 0, keepdims=True)
        
        
        w = w - alpha * grad_w
        
        
    
    return w


def learn_separators(X,y):
    W = []

    classes = np.unique(y)
    for c in classes:
        y_ = np.zeros((len(y),1))
        for i in range(len(y)):
            if c == y[i]:
                y_[i] = 1
            else:
                y_[i] = -1
        a = softmax_grad(X, y_)
        #print(a)
        W.append(a)
    #print(W)
    W = np.asarray(W)
    W.shape = (np.shape(W)[0],np.shape(W)[1])
    W = W.T
    #print(W)
    return W

def accuracy_score(X, y, W, x_):
    # fill here
    pred = 0
    
    new = []
    
    for j in range(4):
        #prevnewpred = newpred

        w_ = np.zeros((2,1))
        b = W[0][j]
        w_[0] = W[1][j]
        w_[1] = W[2][j]

        newpred = b + x_ @ w_
        new.append(newpred)
        #for i in range(len())
        #if pred < newpred:
        #    pred = +1
    new = np.asarray(new)
    new = new.reshape(-1,new.shape[1]).T

    test = np.zeros((40,1))
    korrekt = 0
    for i in range(len(new)):
        test[i] = np.argmax(new[i]) + 1
        if test[i] == y[i]:
            korrekt += +1
    #print(test)
    
    accuracy = korrekt/len(y)
    
    return accuracy
    



def plot_all(X,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.
    X = X.T
    num_classes = np.size(np.unique(y))
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,3,facecolor = 'white',figsize = (10,3))
    for a in range(0,3):
        for i in range(0,num_classes):
            s = np.argwhere(y == i+1)
            s = s[:,0]
            axs[a].scatter(X[1,s],X[2,s], s = 30,color = color_opts[i,:])

        # dress panel correctly
        axs[a].set_xlim(0,1)
        axs[a].set_ylim(0,1)
        axs[a].axis('off')

    r = np.linspace(0,1,150)
    for i in range(0,num_classes):
        z = -W[0,i]/W[2,i] - W[1,i]/W[2,i]*r
        axs[1].plot(r,z,'-k',linewidth = 2,color = color_opts[i,:])

    # fuse individual subproblem separators into one joint rule
    r = np.linspace(0,1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((np.ones((np.size(s),1)),s,t),1)
    f = np.dot(W.T,h.T)
    z = np.argmax(f,0)
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    for i in range(0,num_classes + 1):
        axs[2].contour(s,t,z,num_classes-1,colors = 'k',linewidths = 2.25)



X ,y ,x_  = load_data('four_class_data.csv')

# learn all C vs notC separators
W = learn_separators(X,y)

# calculate accuracies for both training and testing sets
accuracy = accuracy_score(X, y, W, x_)

print("Accuracy: ", accuracy)

# plot data and each subproblem 2-class separator
plot_all(X, y, W)


from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import sklearn.datasets as ds

from sklearn.model_selection import train_test_split



digit_data = ds.load_digits()
print("Number of samples: ", digit_data.data.shape[0])
print("Number of attributes: ", digit_data.data.shape[1])
print("Classes: ", digit_data.target_names)

c = digit_data.images[0]
for i in range(1, 10):
    c = np.concatenate((c, digit_data.images[i]), 1)

plt.gray() 
plt.matshow(c)
plt.show()



X, y = digit_data.data, digit_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state=20)

print("Number of training samples: ", X_train.shape[0])
print("Number of testing samples: ", X_test.shape[0])





def train(X, y):
    
    classifier = SVC(gamma="auto", kernel="linear")
    classifier = classifier.fit(X,y)
    
    preds = classifier.predict(X)
    
    train_accuracy = np.mean(preds==y)
    
    return classifier, train_accuracy

def test(classifier, X, y):
    
    test_accuracy = np.mean(classifier.predict(X) == y)
    
    return test_accuracy



# SVM classifier without feature selection
svm, train_acc = train(X_train, y_train)
test_acc = test(svm, X_test, y_test)

print("Train acc: ", train_acc)
print("Test acc: ", test_acc)


# SVM classifier with RFE
import pandas as pd

def reduceData(X,y):
    estimator = SVC(gamma="auto", kernel="linear")
    selector = RFE(estimator, step=1)
    selector = selector.fit(X_train, y_train)
    count = 0
    abc = []
    for i in range(len(selector.support_)):
        if selector.support_[i] == True:
            count +=1
        else:
            abc.append(selector.ranking_[i])

    #print(X_train)
    #print(X_train.shape)
    print('number of selected features: ',count)
    #print(abc)

    X_train2 = pd.DataFrame(X_train)
    #print(X_train2)
    for i in abc:
        X_train2 = X_train2.drop(i, axis=1)
    X_train2 = X_train2.to_numpy()
    #print(X_train2)
    
    return X_train2, abc
        
x_train, abc = reduceData(X_train, y_train)
x_test = pd.DataFrame(X_test)

for i in abc:
    x_test = x_test.drop(i, axis=1)
x_test = x_test.to_numpy()

svm, train_acc = train(x_train, y_train)
test_acc = test(svm, x_test, y_test)

print("Train acc: ", train_acc)
print("Test acc: ", test_acc)
print("The deleted features are: ",abc)




# PCA for dimensionality reduction
k = 2
(n, m) = X_train.shape
features = X_train.T
Sigma = np.cov(features)

values, vectors = np.linalg.eig(Sigma)
print(values.shape,vectors.shape)

valuesTot = np.sum(values)
var_i = np.array([np.sum(values[: i + 1]) /                     valuesTot * 100.0 for i in range(n)])


vectors_reduced = vectors[ : , :k]
Z = X_train.dot(vectors_reduced)
#print(Z.shape)
U = vectors_reduced
#x,y = np.split(Z,2,1)
print(Z.shape)
classifier, train_accuracy = train(Z,y_train) 



(n, m) = X_test.shape
features = X_test.T
Sigma = np.cov(features)
# Compute eigenvectors and eigenvalues of Sigma
values, vectors = np.linalg.eig(Sigma)
valuesTot = np.sum(values)
var_i = np.array([np.sum(values[: i + 1]) /                     valuesTot * 100.0 for i in range(n)])
vectors_reduced = vectors[ : , :k]
Z1 = X_test.dot(vectors_reduced)

test_acc = test(classifier, Z1, y_test)
print('%.2f %% training accuracy in %d dimensions' % (train_accuracy, k))
print(Z1.shape)
print(test_acc)

plt.figure(figsize=(9,8))
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'black', 6:'brown',7:'orange',8:'pink'}

for label in np.unique(y_train):
    #print(label.dtype)
    #print(Z)
    #print(colors[label])
    ix = np.where(y_train == label)
    #print(ix.dtype)
    #print(Z[ix,0].dtype)
    plt.scatter(Z[ix,0], Z[ix,1], c = colors[label])
plt.legend()
plt.show()
