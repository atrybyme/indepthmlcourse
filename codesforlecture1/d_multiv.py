import numpy as np
from sklearn.datasets import make_regression as generator
import matplotlib.pyplot as plt
def data(x):
    d1 = generator(n_samples=x,n_features=5,n_informative=3,noise=0)
    d2 = {}

    d2[0] = d1[0][0:x,0]
    d2[1] = d1[0][0:x,1]
    d2[2] = d1[0][0:x,2]
    d2[3] = d1[0][0:x,3]
    d2[4] = d1[0][0:x,4]
    d2[5] = d1[1][0:x]
    return(d2)

#plt.plot(d2[3],d2[5],'bo')
#plt.show()
