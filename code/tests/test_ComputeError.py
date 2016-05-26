"""
Project: TC2
Description: test ComputeError.py
"""

import numpy as np
from TC2.utils import ComputeError as ce

X1 = np.zeros([2,2,2])
X2 = np.ones([2,2,2])
X3 = np.random.uniform(size = 8).reshape([2,2,2])

print ce.tfL2Error(X1, X2)
print ce.tfL2Error(X1, X3)


X = np.array([[1,2],[3,4]])
Y = np.array([[1,2],[2,np.nan]])
print ce.npL2Error(X, Y)

X4 = np.random.uniform(size = 24).reshape([2,3,4])
X5 = np.random.uniform(size = 24).reshape([2,3,4])
print ce.npPCT_mode(X4, X5)

X4[0,:,:] = np.nan
X4[1,1,1] = np.nan
print ce.npPCT_mode(X4, X5)
print ce.npAUC(X4, X5)

X6 = np.random.uniform(size = 24).reshape([4,3,2])
X6[:,1,1] = np.nan
X7 = np.random.uniform(size = 24).reshape([4,3,2])
X7[0,0,0] = np.nan

print ce.npGCP_allCell(X6, X7)

X8 = np.random.uniform(size = 20)
X8[0:10] = -X8[0:10]
print np_quantileTo01(X8, top_quantile = 0.2)