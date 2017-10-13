from alfa import data_gen
import numpy as np
#from numpy.matrix import transpose
import matplotlib.pyplot as plt
def cost(x,y,theta):
    x_ = np.matrix(x)
    t = np.matrix(theta)
    #print(x)
    #print(len(y[0]))
    #y_ = np.ndarray(y)
    h = (np.dot(t,x_))
    cst = (np.square(h-y[0]))/(2*len(y[0]))
    return np.sum(cst)
def cost_del(x,y,theta):
        x_ = np.matrix(x)
        t = np.matrix(theta)
        h = (np.dot(t,x_))

        d = (h-y[0])/(len(y[0]))
        cst= np.dot(x_,d.transpose())
        #print(x_)
        #print(d)
        #print(cst)
        cst_d = cst.transpose()
        return np.array(cst_d)




#importing data
tr_dat = data_gen(100)
x_tr =tr_dat[0]
y_tr = tr_dat[1]
x_r=np.ones(100)
alfa = 0.000001 #learning rate
#print(x_r)
#print(len(x_tr[0]))
print([x_r])
print([x_tr[0]])
x_r = np.append([x_r],[np.array(x_tr[0])],axis=0)
#print(x_r)
#x_r = np.append(x_r,[np.multiply((np.sin(x_tr[0])),(x_tr[0]))],axis=0)
#print(x_r)
x_r = np.append(x_r,[np.square(x_tr[0])],axis=0)
theta = np.ones(len(x_r[:,0]))
print(len(theta))
print("initial cost")
print(cost(x_r,y_tr,theta))
for i in range(1000):
    theta = theta - (alfa)*(cost_del(x_r,y_tr,theta)[0])
    plt.plot(i,(cost(x_r,y_tr,theta)),'go')
plt.show()
print("final cost")
print(cost(x_r,y_tr,theta))
plt.plot(x_tr,y_tr,'ro')
plt.plot(np.sort(x_tr[0]),np.sort(np.dot(theta,x_r)))
plt.show()
print(theta)
