import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def make_colorspec(w_hist):
    # make color range for path
    s = np.linspace(0,1,len(w_hist[:round(len(w_hist)/2)]))
    s.shape = (len(s),1)
    t = np.ones(len(w_hist[round(len(w_hist)/2):]))
    t.shape = (len(t),1)
    s = np.vstack((s,t))
    colorspec = []
    colorspec = np.concatenate((s,np.flipud(s)),1)
    colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
    return colorspec

def draw_weight_path(ax,w_hist,g):
    # make colors for plot
    colorspec = make_colorspec(w_hist)
    
    arrows = True

    ### plot function decrease plot in right panel
    for j in range(len(w_hist)):  
        w_val = w_hist[j]

        # plot each weight set as a point
        ax.scatter(w_val[0],w_val[1],g(w_val),s = 80,color = colorspec[j],edgecolor = 'black',linewidth = 2*math.sqrt((1/(float(j) + 1))),zorder = 3)
        
        # plot connector between points for visualization purposes
        if j > 0:
            pt1 = np.array([w_hist[j-1][0], w_hist[j-1][1], np.squeeze(g(w_hist[j-1]))])
            pt2 = np.array([w_hist[j][0], w_hist[j][1], np.squeeze(g(w_hist[j]))])
            
            # if points are different draw error
            if np.linalg.norm(pt1 - pt2) > 0.1 and arrows == True:
                
                ax.plot([pt1[0],pt2[0]], [pt1[1], pt2[1]],[pt1[2],pt2[2]] ,  color ='black',linewidth = 2, zorder=2)
                

def plotFunction(g, w_history=None, view=(50,80),w_min=-1.0,w_max=1.0):
    x = np.linspace(w_min, w_max, 100)
    y = np.linspace(w_min, w_max, 100)

    X, Y = np.meshgrid(x, y)

    Z = np.empty([100,100])
    vectorize_w = lambda w0, w1: np.array([[w0], [w1]])
    for i in range(100):
        for j in range(100):
            Z[i,j] = np.squeeze(g(vectorize_w(X[i,j], Y[i,j])))
    
    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(view[0], view[1])
    
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
    
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('g(w)')
    ax.set_title('Function surface');
    
    if w_history:
        draw_weight_path(ax, w_history, g)


# We define the function 1
g = lambda w: -np.cos(2*np.pi*np.sum(w**2)) + 2 * np.sum(w**2)

# Now, we can call the plotFunction to plot surface plot
plotFunction(g)




def gradientDescent_func1(g, lr = 1e-1, w_0 = None):
    '''
        lr: learning rate
        w_0: initial w
    '''
    #Initialize variables
    w = w_0
    w_history = [w] #List of w's in each iteration, used to plot

    gradient = lambda w: 4*np.pi*w*np.sin(2*np.pi*np.sum(w**2)) + 4*w

    while True:
        
        w_prev = w
        #Update w
        w = w - lr*gradient(w)

        if np.sqrt(np.sum((w_prev - w)**2)) <= 1e-02:
            break

        w_history.append(w)

        
    return w, w_history



w, w_history = gradientDescent_func1(g, lr=0.1, w_0=np.array([-0.6 , -0.6]))

print("Number of iterations: ", len(w_history)-1) #If we did n iterations we will have n+1 w values in history. 
print("Minimum cost: ", g(w))
plotFunction(g, w_history, view=(65,75))



g = lambda w: (w.T @ np.array([[1,0.5], [0.5,1]])) @ w + np.array([[1,1]]) @ w


plotFunction(g,view=(40,-40),w_min=-10, w_max=10)




def NewtonsMethod_func2(g, w_0 = None):
    '''
        w_0: initial w
    '''
    #Initialize variables
    epsilon = 1e-6 # Add it to denominator of update rule to avoid division by zero 
    
    w = w_0
    w_history = [w] #List of w's in each iteration, used to plot

    grad_1st = lambda w: np.array([2*w[0] + w[1] + 1, w[0] + 2*w[1] +1])
    H = np.array([[2,1], [1,2]]) 
    inv_H = np.linalg.inv(H)

    while True:
        
        w_prev = w
        #Update w
        w = w - inv_H@grad_1st(w)
        
        if np.sqrt(np.sum((w_prev - w)**2)) <= 1e-02:
            break            
        #Add new w to list
        w_history.append(w)  
        
    return w, w_history



w, w_history = NewtonsMethod_func2(g, w_0=np.array([[9],[9]]))

# Squeeze w vectors for plotting
for i in range(len(w_history)):
    w = w_history[i]
    w_history[i] = np.squeeze(w)

plotFunction(g, w_history, view=(40,-40),w_min=-10, w_max=10)
print("Number of iterations: ", len(w_history)-1) #If we did n iterations we will have n+1 w values in history. 
print("Minimum cost: ", g(w))




def gradientDescent_func2(g, lr = 0.1, w_0 = None):
    '''
        lr: learning rate
        w_0: initial w
    '''
    #Initialize variables
    w = w_0
    w_history = [w] #List of w's in each iteration, used to plot


    # Code the gradient(w) function here.
    gradient = lambda w: np.array([2*w[0] + w[1] + 1, w[0] + 2*w[1] +1])
    
    #Gradient Descent Steps until the stopping criteria is met
    while True:
        w_prev = w
        #Update w
        w = w - lr*gradient(w)
        
        #Add new w to list
        if np.sqrt(np.sum((w_prev - w)**2)) <= 1e-02:
            break
        
        w_history.append(w)
                
    return w, w_history



w, w_history = gradientDescent_func2(g, w_0=np.array([[1],[1]]))

# Squeeze w vectors for plotting
for i in range(len(w_history)):
    w = w_history[i]
    w_history[i] = np.squeeze(w)

plotFunction(g, w_history, view=(40,-40),w_min=-10, w_max=10)

print("Number of iterations: ", len(w_history)-1) #If we did n iterations we will have n+1 w values in history. 
print("Minimum cost: ", g(w))


# load the data
def load_data(csvname):
    data = np.asarray(pd.read_csv(csvname,header = None))
    x = data[:,0]
    x.shape = (np.size(x),1)
    temp = np.ones((np.size(x),1))
    X = np.concatenate((temp,x),1)
    y = data[:,1]
    y = y/y.max()
    y.shape = (np.size(y),1)
    return X,y



# run gradient descent
def gradient_descent(X,y,w0):
    w_path = []                 # container for weights learned at each iteration
    cost_path = []              # container for associated objective values at each iteration
    w_path.append(w0)
    cost = compute_cost(w0)
    cost_path.append(cost)
    w = w0

    # start gradient descent loop
    max_its = 5000
    alpha = 10**(-2)
    for k in range(max_its):
        # compute gradient
        sig = 1/(1 + my_exp(-np.dot(X,w)))  #Output of sigmoid
        r_k = 2*(sig-y) * sig * (1-sig) #Calculate r^k
        grad = np.matmul(np.transpose(X), r_k) #Calculate logistic regression gradient 
        
        # take gradient step
        w = w - alpha*grad
        
        # take gradient step
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        cost = compute_cost(w)
        cost_path.append(cost)

    # reshape containers for use in plotting in 3d
    w_path = np.asarray(w_path)
    w_path.shape = (np.shape(w_path)[0],np.shape(w_path)[1])
    return w_path,cost_path

# calculate the cost value for a given input weight w
def compute_cost(w):
    temp = 1/(1 + my_exp(-np.dot(X,w))) - y
    temp = np.dot(temp.T,temp)
    return temp[0][0]

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u


# used by plot_logistic_surface to make objective surface of logistic regression cost function
def add_layer(a,b,c):
    a.shape = (2,1)
    b.shape = (1,1)
    z = my_exp(-np.dot(c,a))
    z = 1/(1 + z) - b
    z = z**2
    return z

# plot fit to data and corresponding gradient descent path onto the logistic regression objective surface
def show_fit(w_path,ax,col):
    # plot solution of gradient descent fit to original data
    s = np.linspace(0,25,100)
    t = 1/(1 + my_exp(-(w_path[-1][0] + w_path[-1][1]*s)))
    ax.plot(s,t,color = col)

# plot gradient descent paths on cost surface
def show_paths(w_path,cost_path,ax,col):           
    # plot grad descent path onto surface
    ax.plot(w_path[:,0],w_path[:,1],cost_path,color = col,linewidth = 5)   # add a little to output path so its visible on top of the surface plot
    
# plot logistic regression surface
def plot_surface(ax):
    # plot logistic regression surface
    r = np.linspace(-3,3,100)
    s,t = np.meshgrid(r, r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    # build 3d surface
    surf = np.zeros((np.size(s),1))
    max_its = np.size(y)
    for i in range(0,max_its):
        surf = surf + add_layer(X[i,:],y[i],h)

    # reshape 
    s = np.reshape(s,(100,100))
    t = np.reshape(t,(100,100))
    surf = np.reshape(surf,(100,100))

    # plot 3d surface
    ax.plot_surface(s,t,surf,cmap = 'jet')
    ax.azim = 175
    ax.elev = 20
    
# plot points
def plot_points(X,y,ax):
    ax.plot(X[:,1],y,'ko')


# load dataset
X,y = load_data('bacteria_data.csv') # load in data

# initialize figure, plot data, and dress up panels with axes labels etc.,
fig = plt.figure(facecolor = 'white',figsize = (8,3))
ax1 = fig.add_subplot(121)
ax1.set_xlim(min(X[:,1])-0.5, max(X[:,1])+0.5)
ax1.set_ylim(min(y)-0.1,max(y)+0.1)
ax1.axis('off')

ax2 = fig.add_subplot(122, projection='3d')
ax2.xaxis.set_rotate_label(False)
ax2.yaxis.set_rotate_label(False)
ax2.zaxis.set_rotate_label(False)
ax2.get_xaxis().set_ticks([-3,-1,1,3])
ax2.get_yaxis().set_ticks([-3,-1,1,3])
# ax2.axis('off')

### run gradient descent with first initial point
w0 = np.array([0,2])
w0.shape = (2,1)
w_path, cost_path = gradient_descent(X,y,w0)

# plot points
plot_points(X,y,ax1)

# plot fit to data and path on objective surface
show_fit(w_path,ax1,'m')
show_paths(w_path,cost_path,ax2,'m')

### run gradient descent with first initial point
w0 = np.array([0,-2])
w0.shape = (2,1)
w_path, cost_path = gradient_descent(X,y,w0)

# plot fit to data and path on objective surface
show_fit(w_path,ax1,'c')
show_paths(w_path,cost_path,ax2,'c')
plot_surface(ax2)
plt.show()



# load the data
def load_data(csvname):
    data = np.asarray(pd.read_csv(csvname,header = None))
    x = data[:,0]
    x.shape = (np.size(x),1)
    temp = np.ones((np.size(x),1))
    X = np.concatenate((temp,x),1)
    y = data[:,1]
    y = y/y.max()
    y.shape = (np.size(y),1)
    return X,y


# run gradient descent
def gradient_descent(X,y,w0,lam):
    w_path = []                 # container for weights learned at each iteration
    cost_path = []              # container for associated objective values at each iteration
    w_path.append(w0)
    cost = compute_cost(w0)
    cost_path.append(cost)
    w = w0

    # start gradient descent loop
    max_its = 5000
    alpha = 10**(-2)
    for k in range(max_its):
        # compute gradient
        sig = 1/(1 + my_exp(-np.dot(X,w)))  #Output of sigmoid
        r_k = 2*(sig-y) * sig * (1-sig) #Calculate r^k
        grad = np.matmul(np.transpose(X), r_k) #Calculate logistic regression gradient
        
        #Add regularization gradient
        w_temp = np.copy(w) #Copy w weights
        w_temp[0,0] = 0 #Change the weight of b to 0
        grad += 2*lam*w_temp #Add l2 regularization

        # take gradient step
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        cost = compute_cost(w)
        cost_path.append(cost)

    # reshape containers for use in plotting in 3d
    w_path = np.asarray(w_path)
    w_path.shape = (np.shape(w_path)[0],np.shape(w_path)[1])
    return w_path,cost_path

# calculate the cost value for a given input weight w
def compute_cost(w):
    temp = 1/(1 + my_exp(-np.dot(X,w))) - y
    temp = np.dot(temp.T,temp)
    return temp[0][0]

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u



# used by plot_logistic_surface to make objective surface of logistic regression cost function
def add_layer(a,b,c):
    a.shape = (2,1)
    b.shape = (1,1)
    z = my_exp(-np.dot(c,a))
    z = 1/(1 + z) - b
    z = z**2
    return z

# plot fit to data and corresponding gradient descent path onto the logistic regression objective surface
def show_fit(w_path,ax,col):
    # plot solution of gradient descent fit to original data
    s = np.linspace(0,25,100)
    t = 1/(1 + my_exp(-(w_path[-1,0] + w_path[-1,1]*s)))
    ax.plot(s,t,color = col)

# plot gradient descent paths on cost surface
def show_paths(w_path,cost_path,ax,col):           
    # plot grad descent path onto surface
    ax.plot(w_path[:,0],w_path[:,1],cost_path,color = col,linewidth = 5)   # add a little to output path so its visible on top of the surface plot
    
# plot logistic regression surface
def plot_surface(ax,lam):
    # plot logistic regression surface
    r = np.linspace(-3,3,100)
    s,t = np.meshgrid(r, r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    # build 3d surface
    surf = np.zeros((np.size(s),1))
    max_its = np.size(y)
    for i in range(0,max_its):
        surf = surf + add_layer(X[i,:],y[i],h)
    surf = surf + lam*t**2

    s = np.reshape(s,(100,100))
    t = np.reshape(t,(100,100))
    surf = np.reshape(surf,(100,100))
    
    # plot 3d surface
    ax.plot_surface(s,t,surf,cmap = 'jet')
    ax.azim = 175
    ax.elev = 20
    
# plot points
def plot_points(X,y,ax):
    ax.plot(X[:,1],y,'ko')


# load dataset
X,y = load_data('bacteria_data.csv') # load in data

# initialize figure, plot data, and dress up panels with axes labels etc.,
fig = plt.figure(facecolor = 'white',figsize = (8,3))
ax1 = fig.add_subplot(121)
ax1.set_xlim(min(X[:,1])-0.5, max(X[:,1])+0.5)
ax1.set_ylim(min(y)-0.1,max(y)+0.1)
ax1.axis('off')

ax2 = fig.add_subplot(122, projection='3d')
ax2.xaxis.set_rotate_label(False)
ax2.yaxis.set_rotate_label(False)
ax2.zaxis.set_rotate_label(False)
ax2.get_xaxis().set_ticks([-3,-1,1,3])
ax2.get_yaxis().set_ticks([-3,-1,1,3])
# ax2.axis('off')

# define regularizer parameter
lam = 10**-1

### run gradient descent with first initial point
w0 = np.array([0,2])
w0.shape = (2,1)
w_path, cost_path = gradient_descent(X,y,w0,lam)

# plot points
plot_points(X,y,ax1)

# plot fit to data and path on objective surface
show_fit(w_path,ax1,'m')
show_paths(w_path,cost_path,ax2,'m')

### run gradient descent with first initial point
w0 = np.array([0,-2])
w0.shape = (2,1)
w_path, cost_path = gradient_descent(X,y,w0,lam)

# plot fit to data and path on objective surface
show_fit(w_path,ax1,'c')
show_paths(w_path,cost_path,ax2,'c')
plot_surface(ax2,lam)
plt.show()