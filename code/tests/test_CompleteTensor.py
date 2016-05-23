
import numpy as np
from TC2 import CompleteTensor as ct

X1 = np.zeros([2,2,2])
#X2 = np.ones([2,2,2])
#X3 = np.random.uniform(size = 8).reshape([2,2,2])

paras = {}
methods = ['mean1d', 'mean2d', 'FaLRTC']
for method in methods:
    print ct.CompleteTensor(X1, method, paras)
    
ct.CompleteTensor(X1, 'whatever', paras)