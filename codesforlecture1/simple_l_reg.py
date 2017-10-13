##import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression as generator
##generate data with with certain randomness which can be fitted by simple linear regression.
## function return a matrix with 1st and 2nd column as training set while 3rd and 4th column as training set
def data():
    d1 = generator(n_samples=1500,n_features=1,n_informative=1,noise=13)
    d2 = {}

    d2[0] = d1[0][0:1500,0]
    d2[1] = d1[1][0:1500]
    return(d2)
##visualise the data
def visualise(x,y):
    plt.plot(x,y,'ro')
    plt.show()
def cost(x,y,theta1,theta2):
    h = theta1 +(theta2*x)
    j =(np.subtract(h,y))
    k = np.square(j)
    k = k/(2*len(y))
    return np.sum(k)
def cost_del(x,y,theta1,theta2):
    h = theta1 +(theta2*x)
    j = (np.subtract(h,y))/len(y)
    k={}
    k[0] = np.sum(j)
    k[1] = np.sum(np.multiply(j,x))
    return k
##learning rate
alfa = 0.001
##extract training data
tr = data()
tr[0].size
x_tr =tr[0][0:1400]
y_tr=tr[1][0:1400]
#visualize the training data
visualise(x_tr,y_tr)
#testing data
x_tst = tr[0][1400:1500]
y_tst = tr[1][1400:1500]
theta0 = 0
theta1 = 0
print("initial cost:")
print(cost(x_tr,y_tr,theta0,theta1))
##YOUR CODE HERE:
for i in range(3000):
    theta0 = theta0 - (alfa)*(cost_del(x_tr,y_tr,theta0,theta1)[0])
    theta1 = theta1 - (alfa)*(cost_del(x_tr,y_tr,theta0,theta1)[1])
    plt.plot(i,(cost(x_tr,y_tr,theta0,theta1)),'bo')

## Lets check your code on on test set
print("optimized cost")
print(cost(x_tst,y_tst,theta0,theta1))
plt.show()
plt.plot(x_tr[0:1500],(theta0 +theta1*x_tr)[0:1500], 'ro')
plt.plot(x_tst,y_tst,'bo')
print(theta0)
print(theta1)
plt.show()
