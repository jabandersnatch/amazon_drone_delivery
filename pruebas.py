# Do this (new version)
from numpy.random import default_rng
import numpy as np
import random as rn
x = np.arange(0, 10)
tamano=x[0]
tamano-=1
np.random.shuffle(x)
x=x[:tamano]
arr_combinations=[[[0,0],[0,0]],[[0,0],[0,0]]]
i=sum(arr_combinations)
x = np.reshape(x, (2,5))

x[-1]=x[-1][0:len(tamano)-1]
w=x.tolist()
print(len(x))