
import numpy as np
from TC2 import CrossValidateTensor as cv

#X1 = np.zeros([2,2,2])
#X2 = np.ones([2,2,2])
X3 = np.random.uniform(size = 24).reshape([4,3,2])

X3[0,:,0] = 'nan'
X3[1,:,1] = 'nan'
print X3

paras = {}

cv1 = cv.CV_CompleteTensor('loo fiber', 'mean1d', paras = paras)
X3_CV, runtime = cv1._run(X3)
print X3_CV
print 'run time per fold:' + str(runtime)

cv2 = cv.CV_CompleteTensor('k-fold fiber', 'mean1d', paras = paras)
X3_CV, runtime = cv2._run(X3, kfold = 4)
print X3_CV
print 'run time per fold:' + str(runtime)


cv3 = cv.CV_CompleteTensor('loo fiber', 'mean2d', paras = paras)
X3_CV, runtime = cv3._run(X3)
print X3_CV
print 'run time per fold:' + str(runtime)
