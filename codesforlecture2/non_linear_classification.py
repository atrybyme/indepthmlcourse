# importing importtant libraries
from matplotlib.pyplot import contour
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from numpy import meshgrid
from numpy import arange
# function to visualize data
def visualise(trd1,trd2,y):
    for i in range(569):
        x1 = trd1[i]
        x2 = trd2[i]
        if y[i]==1:
            plt.plot(x1,x2,'ro')
        else:
            plt.plot(x1,x2,'bo')
    plt.show()
def hypothesis(x,theta):
    a = np.dot(theta,x)
    c = -1*a
    d = np.exp(c)
    d = d+1
    b = 1./d
    return b
def grad(x,y,theta):
    h = hypothesis(x,theta)
    a =h-y
    b = np.dot(a,np.transpose(x))
    return b/(len(y))
#load data.
print("Loading Data....\n\n")
data= load_breast_cancer()
tr_data = data.data
tr_res = data.target
data_features = data.feature_names

#Learning rat
alfa = 0.002
# printing feature_names and labels
print("Features available:")
for i in range(30):
    print i,data_features[i]
print("\n\n")

#try with different lables
trd1 = tr_data[:,0]-(np.sum(tr_data[:,0]))/len(tr_res)
trd2 = tr_data[:,1]-(np.sum(tr_data[:,1]))/len(tr_res)
trd = trd1#i have shifted the data around zero by subtracting it with its average
#now we will use x1,x2,x1^2,x2^2,x1*x2 as our input
trd = np.append([trd],[trd2],axis=0)#i have shifted the data around zero by subtracting it with its average
trd = np.append(trd,[np.power(trd1,2)/10],axis=0)# scale them so that they may not create overflow error
trd = np.append(trd,[np.power(trd2,2)/10],axis=0)
trd = np.append(trd,[(trd1)*(trd2)/10],axis=0)
trd = np.append(trd,[np.ones(569)],axis=0)
print(trd)

# #red show melegnin and blue shoes benign
visualise(trd1,trd2,tr_res)
#
#Lets build our model
     #1. initialise theta as a 3 dimensional vector
theta = np.ones(6)/(-10)
hypothesis(trd,theta)
grad(trd,tr_res,theta)
for i in range(6000):
     theta = theta -(alfa)*(grad(trd,tr_res,theta))
     #plot the cost..........
     plt.plot(i,np.sum(np.square((hypothesis(trd,theta))-tr_res)),'ro')
plt.show()
#visualise the plotted parameter
for i in range(569):
    x1 = trd[0,i]
    x2 = trd[1,i]
    if tr_res[i]==1:
        plt.plot(x1,x2,'ro')
    else:
        plt.plot(x1,x2,'bo')
print(theta)
b = hypothesis(trd,theta)
print(np.sum((b-tr_res)*(b-tr_res)))
delta = 0.025
xrange = arange(-10.0, 25.0, delta)
yrange = arange(-10.0, 25.0, delta)
X, Y = meshgrid(xrange,yrange)
F = theta[0]*X + theta[1]*Y +theta[2]*X*X +theta[3]*Y*Y +theta[4]*X*Y+theta[5]
contour(X,Y,F, [0],linewidths=5)
plt.show()
