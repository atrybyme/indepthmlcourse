# importing importtant libraries
from matplotlib.pyplot import contour
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
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
def hypothesis(x1,x2,theta):
    a = theta[0]+(theta[1]*x1) +(theta[2]*x2)
    b = 1/(1+np.exp((-1)*a))
    return b
def grad(x1,x2,y,theta):
    h = hypothesis(x1,x2,theta)
    a =h-y
    b = ((h-y)*x1)/len(y)
    c = ((h-y)*x2)/len(y)
    gr = (np.sum(a))/len(y)
    gr = np.append(gr,np.sum(b))
    gr = np.append(gr,np.sum(c))
    return gr
#load data.
print("Loading Data....\n\n")
data= load_breast_cancer()
tr_data = data.data
tr_res = data.target
data_features = data.feature_names

#Learning rate
alfa = 0.00275
# printing feature_names and labels
print("Features available:")
for i in range(30):
    print i,data_features[i]
print("\n\n")
#try with different lables
trd1 = tr_data[:,0]-(np.sum(tr_data[:,0]))/len(tr_res)
trd2 = tr_data[:,1]-(np.sum(tr_data[:,1]))/len(tr_res)
#red show melegnin and blue shoes benign
visualise(trd1,trd2,tr_res)

#Lets build our model
    #1. initialise theta as a 3 dimensional vector
theta = np.ones(3)
for i in range(5000):
    theta = theta -(alfa)*(grad(trd1,trd2,tr_res,theta))
    plt.plot(i,np.sum(np.square(hypothesis(trd1,trd2,theta) -tr_res)),'go')
plt.show()

#visualise the plotted parameter
for i in range(569):
    x1 = trd1[i]
    x2 = trd2[i]
    if tr_res[i]==1:
        plt.plot(x1,x2,'ro')
    else:
        plt.plot(x1,x2,'bo')

s = trd2[:,None]
contour(trd1, s.ravel(),theta[0]+(theta[1]*trd1) +(theta[2]*s), [0])
plt.show()
