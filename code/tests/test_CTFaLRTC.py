
import numpy as np
import tensorflow as tf
from TC2.completionMethods import CTFaLRTC

X1 = np.zeros([2,2,2])
#X2 = np.ones([2,2,2])
#X3 = np.random.uniform(size = 8).reshape([2,2,2])

paras = {}
sess = tf.Session()
X1_out = CTFaLRTC.FaLRTC.run(X1, paras, sess)
X1_out =  m1._run(sess)
