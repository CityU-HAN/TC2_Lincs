
import numpy as np
from TC2 import CompleteTensor as ct

X1 = np.zeros([2,2,2])
#X2 = np.ones([2,2,2])
X3 = np.random.uniform(size = 24).reshape([4,3,2])
X3[0,:,1] = np.nan
X3[3,:,0] = np.nan
print X3

paras = {}
methods = ['mean1d', 'mean2d', 'FaLRTC']
for method in methods:
    print ct.CompleteTensor(X3, method, paras)

paras = {'alpha': 0.1}
print ct.CompleteTensor(X3, 'mean2d', paras)
    
ct.CompleteTensor(X1, 'whatever', paras)