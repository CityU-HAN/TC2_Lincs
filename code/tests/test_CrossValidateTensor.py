
import numpy as np
from TC2 import CrossValidateTensor as cv

#X1 = np.zeros([2,2,2])
#X2 = np.ones([2,2,2])
X3 = np.random.uniform(size = 8).reshape([2,2,2])

X3[0,0,0] = 'nan'
X3[1,1,1] = 'nan'

paras = {}

cv1 = cv.CV_CompleteTensor('loo fiber', 'mean1d', paras = paras)
X3_CV, runtime = cv1._run(X3)