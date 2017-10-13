from math import *
from numpy.random import rand
import matplotlib.pyplot as plt
def gen(x):
    y=0
    y = y +(1+ x)
    y = y +(14*(x*x)+ 5)
    #y = y+ (2*(x)*sin(x) +1)
    return y
x=[]
y=[]

def data_gen(samples):
    for i in range(samples):
        e = (rand())*30
        x.append(e)
        y.append(gen(e)+ (rand())*400)
        plt.plot(x,y,'ro')
    plt.show()
    dat = [[x],[y]]
    return(dat)
#t = data_gen(150)
